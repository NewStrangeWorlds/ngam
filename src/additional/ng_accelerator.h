#ifndef _ng_accelerator_h
#define _ng_accelerator_h

#include <vector>
#include <cmath>
#include <numeric>


namespace ngam {


class NgAccelerator {
  public:
    NgAccelerator(size_t ng_interval_)
      : interval(ng_interval_) {}

    // Call every iteration. Stores profiles at regular intervals and
    // applies Ng acceleration once 4 profiles have been collected.
    // Returns true (and modifies temperature in-place) when acceleration is applied.
    //
    // Ng (1974) / Olson, Auer & Buchler (1986). With four stored iterates T0..T3 of an (assumed)
    // affine map T_{k+1} = F(T_k), successive differences d_k = T_k - T_{k-1} and second
    // differences D1 = d2 - d1, D2 = d3 - d2, the trial point
    //     y = T2 - c2 d2 - c1 d1        has image      T* = F(y) = T3 - c2 d3 - c1 d2
    // and residual F(y) - y = d3 - c1 D1 - c2 D2. The coefficients minimise that residual in the
    // least-squares sense, and the accelerated profile is T* (no extra forward evaluation needed).
    // NOTE the extrapolation subtracts the FIRST differences d3, d2 from T3. Subtracting the second
    // differences instead (T3 - c1 D1 - c2 D2 = T2 + minimised residual) returns essentially T2,
    // i.e. undoes the last two intervals of progress -- that was the bug fixed on 2026-09-05.
    bool accelerate(std::vector<double>& temperature, size_t iteration)
    {
      if (interval == 0) return false;

      // store a profile every `interval` iterations (starting from iteration 0)
      if (iteration % interval == 0)
        stored.push_back(temperature);

      if (stored.size() < 4) return false;

      // we have T0, T1, T2, T3 — apply Ng acceleration
      const size_t n = temperature.size();
      const auto& T0 = stored[0];
      const auto& T1 = stored[1];
      const auto& T2 = stored[2];
      const auto& T3 = stored[3];

      // successive differences
      std::vector<double> d1(n), d2(n), d3(n);
      for (size_t i = 0; i < n; ++i)
      {
        d1[i] = T1[i] - T0[i];
        d2[i] = T2[i] - T1[i];
        d3[i] = T3[i] - T2[i];
      }

      // changes in differences
      std::vector<double> D1(n), D2(n);
      for (size_t i = 0; i < n; ++i)
      {
        D1[i] = d2[i] - d1[i];
        D2[i] = d3[i] - d2[i];
      }

      // build 2x2 normal equations: A * c = b
      const double a11 = dot(D1, D1);
      const double a12 = dot(D1, D2);
      const double a22 = dot(D2, D2);
      const double b1  = dot(D1, d3);
      const double b2  = dot(D2, d3);

      const double det = a11 * a22 - a12 * a12;

      // if the system is (nearly) singular, skip acceleration (scale-free test: the two
      // second-difference vectors are parallel, e.g. a single dominant mode)
      if (std::abs(det) < 1e-12 * a11 * a22 || a11 == 0 || a22 == 0)
      {
        stored.clear();
        return false;
      }

      const double c1 = (a22 * b1 - a12 * b2) / det;
      const double c2 = (a11 * b2 - a12 * b1) / det;

      // extrapolate: T* = T3 - c2*d3 - c1*d2 (see the derivation above)
      std::vector<double> T_new(n);
      for (size_t i = 0; i < n; ++i)
        T_new[i] = T3[i] - c2 * d3[i] - c1 * d2[i];

      // safety: reject if any temperature is unphysical
      for (size_t i = 0; i < n; ++i)
      {
        if (T_new[i] < 1.0)
        {
          stored.clear();
          return false;
        }
      }

      temperature = T_new;
      stored.clear();
      return true;
    }

  private:
    size_t interval;
    std::vector<std::vector<double>> stored;

    static double dot(const std::vector<double>& a, const std::vector<double>& b)
    {
      double s = 0;
      for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
      return s;
    }
};


}

#endif
