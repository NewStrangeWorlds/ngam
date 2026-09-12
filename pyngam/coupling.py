"""Alternating coupling of an ngam model with a chemical kinetics code (neoVULCAN).

Both codes solve a steady state: ngam the radiative-convective equilibrium for a frozen
composition, the kinetics code the chemical steady state for a frozen atmosphere. The driver
alternates them (Gauss-Seidel) with the kinetics code on its own pressure grid:

    ngam.compute(tol)  ->  T(p), Kzz(p)  ->  kinetics to steady state  ->  mixing ratios
         ^                                                                      |
         +-------- external chemistry module  <--  Aitken relaxation + dead band  <-+

The composition update is under-relaxed with Aitken's dynamic weight on log mixing ratios and
committed only when it exceeds a dead band, for the same reasons as the model's self-consistent
Kzz (see atmosphere/eddy_diffusion.h): the undamped alternation oscillates, and every committed
change costs the Newton its quadratic tail. The RCE tolerance is loosened on the first passes and
tightened by a decade per pass.

The kinetics code is accessed through a small duck-typed interface (neoVULCAN's
``VulcanChemistry`` satisfies it): ``species`` (list of names), ``data_atm.pco`` (pressure grid
in dyn/cm^2, bottom to top), ``set_atmosphere(T=, P=, Kzz=)``, ``set_composition(species, table)``,
``run_to_convergence(warm_start=)``, ``get_mixing_ratios()`` -> (nz, ni), ``get_convergence_info()``.
"""

import contextlib
import os
import time

import numpy as np

from . import mixing_ratios, species_symbols

LN10 = np.log(10.0)


def interpolate_log(p_from, values, p_to, log_values=False):
    """Interpolate a profile in log pressure (constant extension beyond the range).

    ``values`` may be 1D or (n_levels, k); with ``log_values`` the interpolation is done on
    log10 of the values (floored at 1e-300), the natural choice for mixing ratios.
    """
    x_from = np.log10(np.asarray(p_from, dtype=float))
    x_to = np.log10(np.asarray(p_to, dtype=float))
    order = np.argsort(x_from)
    v = np.asarray(values, dtype=float)
    if log_values:
        v = np.log10(np.maximum(v, 1e-300))
    if v.ndim == 1:
        out = np.interp(x_to, x_from[order], v[order])
    else:
        out = np.column_stack([np.interp(x_to, x_from[order], v[order, k]) for k in range(v.shape[1])])
    return 10.0 ** out if log_values else out


class KineticsCoupling:
    """Drive ngam and a kinetics code to a common steady state.

    Parameters
    ----------
    model : pyngam model
        Initialised with a chemistry list ending in ``("external", {})``, normally
        ``[equilibrium, quench, external]`` so the quench approximation seeds the kinetics.
    chem : kinetics code instance (see the module docstring), already initialised.
    species : list of str, optional
        Species exchanged between the codes (default: every name both codes know).
    rce_tolerance : (float, float)
        RCE convergence threshold of the first pass and the final one; tightened by a decade
        per pass in between.
    dead_band : float
        Composition changes below this (dex, over the tracked species and levels) are not
        committed. The kinetics code's own convergence noise is ~0.03 dex, so keep this above it.
    composition_tolerance : float
        The coupling is converged when the raw composition change falls below this (dex) at
        the final RCE tolerance.
    relax : float
        Initial Aitken weight for the composition update (then adapted, clamped to [0.05, 1]).
    tracked_species : list of str, optional
        Species whose change decides relaxation, dead band and convergence. Default: the
        exchanged species that are also opacity sources of the model (from ``model.config``),
        falling back to all exchanged species. Every exchanged species is still handed over;
        this only keeps trace radicals swinging across the floor -- 5 dex swings of H, HCN or
        CH4 near 1e-10 in the photochemical hot-Jupiter test -- from driving the Aitken weight
        to its clamp while the opacity-relevant composition has long settled.
    tracked_floor : float
        Mixing ratios below this (in both old and new composition) are ignored when measuring
        changes: they do not matter for the opacity and their log changes are noise.
    max_passes : int
    seed_continuation : bool
        Run pass 0 in two stages: first with every `quench` module disabled (pure equilibrium),
        then, from that converged profile, with the quench module enabled and a re-solve. Off by
        default. It rescues a strongly IRRADIATED planet started from a cold analytic profile,
        where the quench composition is so far from the converged one (every family quenched at
        the grid bottom, a CH4/NH3 column to the top) that the Newton never recovers: +1700 K at
        the bottom in one call. HELIOS's "force equilibrium chemistry on the first iteration" is
        the same device. It HURTS a self-luminous object: switching the quench module on at a
        converged convective profile is an order-of-magnitude opacity change (CO appears aloft,
        NH3 drops 20x) from which neither the ramped nor the full-strength Newton recovers
        (brown dwarf: stalled at residuals 67 and 15), while the cold quench start converges in
        32 iterations. No effect without a quench module.
    workdir : str, optional
        Directory to switch to around every call into the kinetics code. neoVULCAN reads its
        thermodynamic data relative to its base directory on every rate update, while ngam's
        FastChem parameter file lists paths relative to the ngam directory, so the two cannot
        share a working directory.
    verbose : bool
    """

    def __init__(self, model, chem, species=None, rce_tolerance=(1e-2, 1e-5), dead_band=0.05,
                 composition_tolerance=0.05, relax=0.5, tracked_species=None, tracked_floor=1e-10,
                 max_passes=12, seed_continuation=False, workdir=None, verbose=True):
        self.model = model
        self.chem = chem
        self.workdir = workdir
        self.seed_continuation = seed_continuation
        ngam_species = set(species_symbols())
        self.species = [s for s in (species or chem.species) if s in ngam_species and s in chem.species]
        if not self.species:
            raise ValueError("no species in common between the model and the kinetics code")

        if tracked_species is None:
            opacity = getattr(model, "config", {}).get("opacity_species", [])
            opacity = {(pair[0] if isinstance(pair, (list, tuple)) else pair) for pair in opacity}
            tracked_species = [s for s in self.species if s in opacity] or list(self.species)
        self.tracked = np.array([s in tracked_species for s in self.species])
        if not self.tracked.any():
            raise ValueError("none of the tracked species is exchanged with the kinetics code")
        self.rce_tolerance = rce_tolerance
        self.dead_band = dead_band
        self.composition_tolerance = composition_tolerance
        self.relax = relax
        self.tracked_floor = tracked_floor
        self.max_passes = max_passes
        self.verbose = verbose
        self.history = []

        # Aitken state (on ln mixing ratios, ngam grid)
        self._weight = relax
        self._residual_prev = None

    @contextlib.contextmanager
    def _in_workdir(self):
        if self.workdir is None:
            yield
            return
        previous = os.getcwd()
        os.chdir(self.workdir)
        try:
            yield
        finally:
            os.chdir(previous)

    # ---- grid transfer ------------------------------------------------------------------------
    def _atmosphere_to_kinetics(self):
        atm = self.model.atmosphere
        p_n = np.asarray(atm.pressure)                    # bar, bottom -> top
        p_v = np.asarray(self.chem.data_atm.pco) / 1e6    # bar, bottom -> top
        T_v = interpolate_log(p_n, np.asarray(atm.temperature), p_v)
        p_if = np.sqrt(p_v[1:] * p_v[:-1])                # nz-1 interfaces
        kzz_v = interpolate_log(p_n, np.asarray(atm.kzz), p_if, log_values=True)
        return p_v, T_v, kzz_v

    def _composition_to_kinetics(self):
        p_n = np.asarray(self.model.atmosphere.pressure)
        p_v = np.asarray(self.chem.data_atm.pco) / 1e6
        _, table = mixing_ratios(self.model, self.species)
        return interpolate_log(p_n, np.asarray(table), p_v, log_values=True)

    def _composition_from_kinetics(self):
        p_n = np.asarray(self.model.atmosphere.pressure)
        p_v = np.asarray(self.chem.data_atm.pco) / 1e6
        y_v = np.asarray(self.chem.get_mixing_ratios())
        cols = [self.chem.species.index(s) for s in self.species]
        return interpolate_log(p_v, y_v[:, cols], p_n, log_values=True)

    # ---- composition update ---------------------------------------------------------------------
    def _relaxed_update(self, y_old, y_raw):
        """Aitken-relaxed update on ln y; returns (y_new, max_change_dex, committed, weight)."""
        valid = (y_old > self.tracked_floor) & (y_raw > self.tracked_floor)
        r = np.zeros_like(y_old)
        r[valid] = np.log(y_raw[valid]) - np.log(y_old[valid])
        mask = valid & self.tracked[np.newaxis, :]          # the change metric: tracked species only
        max_change = np.abs(r[mask]).max() / LN10 if mask.any() else 0.0
        if mask.any():
            r_tracked = np.where(mask, np.abs(r), 0.0)
            level, k = np.unravel_index(np.argmax(r_tracked), r.shape)
            self.last_change_at = (self.species[k], int(level))

        if max_change < self.dead_band:
            return y_old, max_change, False, self._weight

        r_aitken = np.where(mask, r, 0.0)
        if self._residual_prev is not None:
            dr = r_aitken - self._residual_prev
            den = float(np.sum(dr * dr))
            if den > 0.0:
                self._weight = -self._weight * float(np.sum(self._residual_prev * dr)) / den
                self._weight = min(1.0, max(0.05, self._weight))
        self._residual_prev = r_aitken

        y_new = np.where(valid, np.exp(np.log(np.maximum(y_old, 1e-300)) + self._weight * r), y_raw)
        return y_new, max_change, True, self._weight

    # ---- the loop --------------------------------------------------------------------------------
    def run(self):
        """Alternate until both codes agree; returns True on convergence."""
        tol_start, tol_final = self.rce_tolerance
        tol = tol_start
        log = self._log

        log(f"exchanging {len(self.species)} species; change metric on "
            f"{', '.join(s for s, t in zip(self.species, self.tracked) if t)}")
        t0 = time.time()
        staged = self.seed_continuation and self.model.set_chemistry_enabled("quench", False) > 0
        if staged:
            log(f"pass 0a: ngam with equilibrium chemistry (quench disabled), tolerance {tol:.1e}")
            self.model.compute(tolerance=tol, warm_start=False)
            self.model.set_chemistry_enabled("quench", True)
            log(f"        done in {time.time()-t0:.0f} s; pass 0b: quench enabled, re-solve")
            t0 = time.time()
            # not a warm start: switching the quench module on changes CO and NH3 aloft by orders
            # of magnitude, and a self-luminous object then needs the mixing-length ramp again
            # (measured on the brown dwarf: warm_start=True stalled pass 0b at a residual of 18)
            self.model.compute(tolerance=tol, warm_start=False)
        else:
            log(f"pass 0: ngam with the seed composition, tolerance {tol:.1e}")
            self.model.compute(tolerance=tol, warm_start=False)
        log(f"        ngam done in {time.time()-t0:.0f} s")

        # seed the kinetics with ngam's composition (equilibrium / quench), on its own grid
        p_v, T_v, kzz_v = self._atmosphere_to_kinetics()
        with self._in_workdir():
            self.chem.set_atmosphere(T=T_v, P=p_v * 1e6, Kzz=kzz_v)
            self.chem.set_composition(self.species, self._composition_to_kinetics())
        _, y_committed = mixing_ratios(self.model, self.species)
        y_committed = np.asarray(y_committed)

        converged = False
        for k in range(1, self.max_passes + 1):
            t0 = time.time()
            with self._in_workdir():
                self.chem.run_to_convergence(warm_start=(k > 1))
            info = self.chem.get_convergence_info()
            y_raw = self._composition_from_kinetics()
            y_new, change, committed, weight = self._relaxed_update(y_committed, y_raw)
            where = getattr(self, "last_change_at", None)
            log(f"pass {k}: kinetics {time.time()-t0:.0f} s, {info.get('count')} steps, "
                f"end_case {info.get('end_case')}; composition change {change:.3f} dex"
                + (f" ({where[0]} at level {where[1]})" if where else "")
                + (f", committed with weight {weight:.2f}" if committed else " (below dead band, not committed)"))

            if committed:
                self.model.set_composition(self.species, y_new.tolist())
                y_committed = y_new

            tol = max(tol / 10.0, tol_final)
            t0 = time.time()
            rce_ok = self.model.compute(tolerance=tol, warm_start=True)
            log(f"        ngam tolerance {tol:.1e}: {'converged' if rce_ok else 'NOT converged'} in {time.time()-t0:.0f} s")

            self.history.append(dict(pass_=k, change_dex=change, committed=committed, weight=weight,
                                     rce_tolerance=tol, rce_converged=rce_ok, kinetics=info))

            if change < self.composition_tolerance and tol <= tol_final and rce_ok:
                converged = True
                log(f"coupling converged after {k} passes")
                break

            p_v, T_v, kzz_v = self._atmosphere_to_kinetics()
            with self._in_workdir():
                self.chem.set_atmosphere(T=T_v, Kzz=kzz_v)

        if not converged:
            log(f"coupling NOT converged after {self.max_passes} passes")
        return converged

    def _log(self, msg):
        if self.verbose:
            print("[coupling] " + msg, flush=True)
