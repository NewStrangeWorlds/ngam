#include "thermodynamic_data.h"
#include "physical_const.h"

#include <map>
#include <cmath>


namespace ngam {


// NASA 7-coefficient polynomial data
// Source: McBride, Zehe, Gordon (2002), NASA/TP-2002-211556
// Format: temp_low, temp_mid, temp_high,
//         low-T coefficients a1-a7,
//         high-T coefficients a1-a7
// c_p/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
static const std::map<chemical_species_id, NASACoefficients> nasa_database = {
  {_H2, {200.0, 1000.0, 6000.0,
    {2.34433112E+00,  7.98052075E-03, -1.94781510E-05,  2.01572094E-08, -7.37611761E-12, -9.17935173E+02,  6.83010238E-01},
    {2.93286575E+00,  8.26608026E-04, -1.46402364E-07,  1.54100414E-11, -6.88804800E-16, -8.13065581E+02, -1.02432865E+00}}},

  {_He, {200.0, 1000.0, 6000.0,
    {2.50000000E+00,  0.0,  0.0,  0.0,  0.0, -7.45375000E+02,  9.28723974E-01},
    {2.50000000E+00,  0.0,  0.0,  0.0,  0.0, -7.45375000E+02,  9.28723974E-01}}},

  {_H, {200.0, 1000.0, 6000.0,
    {2.50000000E+00,  0.0,  0.0,  0.0,  0.0,  2.54736599E+04, -4.46682853E-01},
    {2.50000286E+00, -5.65334214E-09,  3.63251723E-12, -9.19949720E-16,  7.95260746E-20,  2.54736589E+04, -4.46698494E-01}}},

  {_H2O, {200.0, 1000.0, 6000.0,
    {4.19864056E+00, -2.03643410E-03,  6.52040211E-06, -5.48797062E-09,  1.77197817E-12, -3.02937267E+04, -8.49032208E-01},
    {2.67703787E+00,  2.97318160E-03, -7.73769690E-07,  9.44336689E-11, -4.26900959E-15, -2.98858938E+04,  6.88255571E+00}}},

  {_CO, {200.0, 1000.0, 6000.0,
    {3.57953347E+00, -6.10353680E-04,  1.01681433E-06,  9.07005884E-10, -9.04424499E-13, -1.43440860E+04,  3.50840928E+00},
    {3.04848583E+00,  1.35172818E-03, -4.85794075E-07,  7.88536486E-11, -4.69807489E-15, -1.42677477E+04,  6.01709790E+00}}},

  {_CO2, {200.0, 1000.0, 6000.0,
    {2.35677352E+00,  8.98459677E-03, -7.12356269E-06,  2.45919022E-09, -1.43699548E-13, -4.83719697E+04,  9.90105222E+00},
    {4.63659493E+00,  2.74131991E-03, -9.95828531E-07,  1.60373011E-10, -9.16103468E-15, -4.90249341E+04, -1.93534855E+00}}},

  {_CH4, {200.0, 1000.0, 6000.0,
    {5.14987613E+00, -1.36709788E-02,  4.91800599E-05, -4.84743026E-08,  1.66693956E-11, -1.02466476E+04, -4.64130376E+00},
    {1.65326226E+00,  1.00263099E-02, -3.31661238E-06,  5.36483138E-10, -3.14696758E-14, -1.00095936E+04,  9.90506283E+00}}},

  {_NH3, {200.0, 1000.0, 6000.0,
    {4.28648290E+00, -4.66005000E-03,  2.17190502E-05, -2.28063068E-08,  8.26388308E-12, -6.74132170E+03, -6.25371490E-01},
    {2.63455277E+00,  5.66616195E-03, -1.72786975E-06,  2.38681781E-10, -1.23617865E-14, -6.54477910E+03,  6.56632830E+00}}},

  {_H2S, {200.0, 1000.0, 6000.0,
    {4.12023560E+00, -1.57186251E-03,  7.56973740E-06, -6.14698050E-09,  1.77895040E-12, -3.44700640E+03,  2.78550178E+00},
    {2.97833600E+00,  3.60550300E-03, -1.29525060E-06,  2.13015290E-10, -1.28804810E-14, -3.48285350E+03,  6.65399900E+00}}},

  {_N2, {200.0, 1000.0, 6000.0,
    {3.53100528E+00, -1.23660988E-04, -5.02999433E-07,  2.43530612E-09, -1.40881235E-12, -1.04697628E+03,  2.96747038E+00},
    {2.95257637E+00,  1.39690040E-03, -4.92631603E-07,  7.86010195E-11, -4.60755204E-15, -9.23948688E+02,  5.87188762E+00}}},

  {_HCN, {200.0, 1000.0, 6000.0,
    {2.25890802E+00,  1.00510152E-02, -1.33514210E-05,  1.00920804E-08, -3.00888520E-12,  1.48792164E+04,  8.91644378E+00},
    {3.84951793E+00,  3.14164600E-03, -1.06345044E-06,  1.66117340E-10, -9.79924220E-15,  1.45879853E+04,  1.57419080E+00}}},

  {_C2H2, {200.0, 1000.0, 6000.0,
    {8.08681094E-01,  2.33615629E-02, -3.55171815E-05,  2.80152437E-08, -8.50072974E-12,  2.64289807E+04,  1.39397847E+01},
    {4.14756964E+00,  5.96166664E-03, -2.37294852E-06,  4.67412171E-10, -3.61235213E-14,  2.59359886E+04, -1.23028121E+00}}},

  {_OH, {200.0, 1000.0, 6000.0,
    {3.99201543E+00, -2.40131752E-03,  4.61793841E-06, -3.88113333E-09,  1.36411470E-12,  3.61508056E+03, -1.03925458E-01},
    {2.83864607E+00,  1.10725586E-03, -2.93914978E-07,  4.20524247E-11, -2.42169092E-15,  3.94395852E+03,  5.84452662E+00}}},
};


// Monatomic species have c_p/R = 5/2 (ideal gas)
bool ThermodynamicData::isMonatomic(chemical_species_id species)
{
  switch(species) {
    case _H: case _He: case _C: case _O:
    case _Fe: case _Fep: case _Ca: case _Ti: case _Tip:
    case _Na: case _K: case _e:
    case _V: case _Vp: case _Mn: case _Si: case _Cr: case _Crp:
      return true;
    default:
      return false;
  }
}


double ThermodynamicData::evaluateCpOverR(
  const std::array<double, 7>& a, double T)
{
  return a[0] + a[1]*T + a[2]*T*T + a[3]*T*T*T + a[4]*T*T*T*T;
}


double ThermodynamicData::cpOverR(
  chemical_species_id species, double temperature)
{
  // look up in the NASA database
  auto it = nasa_database.find(species);

  if (it != nasa_database.end())
  {
    const auto& data = it->second;

    // clamp temperature to valid range
    double T = temperature;
    if (T < data.temp_low) T = data.temp_low;
    if (T > data.temp_high) T = data.temp_high;

    if (T <= data.temp_mid)
      return evaluateCpOverR(data.low, T);
    else
      return evaluateCpOverR(data.high, T);
  }

  // fallback: monatomic ideal gas
  if (isMonatomic(species))
    return 2.5;

  // unknown species — assume diatomic as a rough default
  return 3.5;
}


double ThermodynamicData::heatCapacityMolar(
  chemical_species_id species, double temperature)
{
  return cpOverR(species, temperature) * constants::gas_constant;
}


double ThermodynamicData::meanHeatCapacity(
  const std::vector<double>& number_densities,
  double temperature)
{
  double n_total = number_densities[_TOTAL];
  if (n_total <= 0) return 0;

  double cp_molar_mean = 0;
  double mu = 0;

  for (const auto& sp : constants::species_data)
  {
    if (sp.id == _TOTAL) continue;

    double x_i = number_densities[sp.id] / n_total;
    if (x_i <= 0) continue;

    cp_molar_mean += x_i * cpOverR(sp.id, temperature);
    mu += x_i * sp.molecular_weight;
  }

  // cp_molar_mean is in units of R, convert to erg/(g·K)
  return cp_molar_mean * constants::gas_constant / mu;
}


double ThermodynamicData::adiabaticGradient(
  const std::vector<double>& number_densities,
  double temperature)
{
  double n_total = number_densities[_TOTAL];
  if (n_total <= 0) return 0;

  double cp_molar_mean = 0;

  for (const auto& sp : constants::species_data)
  {
    if (sp.id == _TOTAL) continue;

    double x_i = number_densities[sp.id] / n_total;
    if (x_i <= 0) continue;

    cp_molar_mean += x_i * cpOverR(sp.id, temperature);
  }

  // nabla_ad = R / c_p_molar = 1 / (c_p/R)
  if (cp_molar_mean <= 0) return 0;

  return 1.0 / cp_molar_mean;
}


}
