/*
* This file is part of the ngam code.
* Copyright (C) 2026 Daniel Kitzmann
*
* ngam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef _module_params_h
#define _module_params_h

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../additional/exceptions.h"


namespace ngam {


// A pluggable component (chemistry module, radiative-transfer solver, convection scheme,
// temperature-correction scheme, ...) is configured by a *module spec*: the module's type keyword
// plus a flat map of NAMED parameters. Parameters arrive as strings (this is what a config file or
// the Python layer hands over) and are converted by the selector that consumes them, which also
// rejects keys it does not understand -- so a knob that belongs to one scheme can never be passed
// silently to another.
using ModuleParams = std::map<std::string, std::string>;

struct ModuleSpec {
  std::string type;
  ModuleParams params;

  ModuleSpec() = default;
  ModuleSpec(std::string type_, ModuleParams params_ = {})
    : type(std::move(type_)), params(std::move(params_)) {}
};


// Typed, validated access to a ModuleSpec's parameters. Every key a selector asks for -- present or
// not -- is recorded as "known"; finish() then rejects any key the caller supplied that was never
// asked for, and lists the known ones in the error. Selectors call finish() once they have read
// everything they accept.
class ParamReader {
  public:
    ParamReader(const ModuleSpec& spec_, std::string context_)
      : spec(spec_), context(std::move(context_)) {}

    bool has(const std::string& key)
    {
      known.insert(key);
      return spec.params.count(key) > 0;
    }

    std::string getString(const std::string& key, const std::string& default_value)
    {
      return has(key) ? spec.params.at(key) : default_value;
    }

    std::string requireString(const std::string& key)
    {
      if (!has(key)) missing(key);
      return spec.params.at(key);
    }

    double getDouble(const std::string& key, const double default_value)
    {
      return has(key) ? toDouble(key, spec.params.at(key)) : default_value;
    }

    double requireDouble(const std::string& key)
    {
      if (!has(key)) missing(key);
      return toDouble(key, spec.params.at(key));
    }

    long getInt(const std::string& key, const long default_value)
    {
      return has(key) ? toInt(key, spec.params.at(key)) : default_value;
    }

    long requireInt(const std::string& key)
    {
      if (!has(key)) missing(key);
      return toInt(key, spec.params.at(key));
    }

    bool getBool(const std::string& key, const bool default_value)
    {
      return has(key) ? toBool(key, spec.params.at(key)) : default_value;
    }

    // all keys the caller supplied; marks every one of them known (for modules such as the
    // isoprofile chemistry whose keys are data, not a fixed vocabulary)
    std::vector<std::string> allKeys()
    {
      std::vector<std::string> keys;
      for (const auto& kv : spec.params)
      {
        keys.push_back(kv.first);
        known.insert(kv.first);
      }
      return keys;
    }

    // reject parameters that no accessor asked for
    void finish()
    {
      std::vector<std::string> unknown;
      for (const auto& kv : spec.params)
        if (known.count(kv.first) == 0) unknown.push_back(kv.first);

      if (unknown.empty()) return;

      std::ostringstream msg;
      msg << "Unknown parameter" << (unknown.size() > 1 ? "s" : "") << " for " << context
          << " '" << spec.type << "': ";
      for (size_t i = 0; i < unknown.size(); ++i)
        msg << (i ? ", " : "") << "'" << unknown[i] << "'";
      msg << ". Accepted parameters: ";
      if (known.empty()) msg << "(none)";
      size_t i = 0;
      for (const auto& k : known) msg << (i++ ? ", " : "") << k;
      msg << "\n";
      throw InvalidInput(context, msg.str());
    }

    const ModuleSpec& spec;

  private:
    std::string context;
    std::set<std::string> known;

    [[noreturn]] void missing(const std::string& key)
    {
      throw InvalidInput(context,
        "Required parameter '" + key + "' missing for " + context + " '" + spec.type + "'\n");
    }

    double toDouble(const std::string& key, const std::string& value)
    {
      try {
        size_t consumed = 0;
        const double v = std::stod(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return v;
      }
      catch (const std::exception&) {
        throw InvalidInput(context,
          "Parameter '" + key + "' of " + context + " '" + spec.type
          + "' must be a number, got '" + value + "'\n");
      }
    }

    long toInt(const std::string& key, const std::string& value)
    {
      const double v = toDouble(key, value);
      if (v != static_cast<double>(static_cast<long>(v)))
        throw InvalidInput(context,
          "Parameter '" + key + "' of " + context + " '" + spec.type
          + "' must be an integer, got '" + value + "'\n");
      return static_cast<long>(v);
    }

    bool toBool(const std::string& key, const std::string& value)
    {
      std::string v = value;
      std::transform(v.begin(), v.end(), v.begin(), ::tolower);
      if (v == "true" || v == "1" || v == "yes" || v == "on") return true;
      if (v == "false" || v == "0" || v == "no" || v == "off") return false;
      throw InvalidInput(context,
        "Parameter '" + key + "' of " + context + " '" + spec.type
        + "' must be a boolean, got '" + value + "'\n");
    }
};


// Resolve a type keyword against a module's list of long names and (optional) short aliases;
// returns the index or throws listing the valid names.
inline size_t resolveModuleType(
  const std::string& type,
  const std::vector<std::string>& names,
  const std::vector<std::string>& short_names,
  const std::string& context)
{
  auto it = std::find(names.begin(), names.end(), type);
  if (it != names.end()) return std::distance(names.begin(), it);

  auto it_short = std::find(short_names.begin(), short_names.end(), type);
  if (it_short != short_names.end()) return std::distance(short_names.begin(), it_short);

  std::string msg = context + " type '" + type + "' unknown! Available: ";
  for (size_t i = 0; i < names.size(); ++i) msg += (i ? ", " : "") + names[i];
  throw InvalidInput(context, msg + "\n");
}


}
#endif
