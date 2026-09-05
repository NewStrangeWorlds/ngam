"""Run a model from a config file:  python -m pyngam config.toml [-o output.nc]"""

import argparse

from .config import load_config, run


def main():
    parser = argparse.ArgumentParser(
        prog="python -m pyngam",
        description="Run an ngam model from a TOML / YAML / JSON config file.")
    parser.add_argument("config", help="config file")
    parser.add_argument("-o", "--output", help="output file (overrides [output].file)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output:
        config.setdefault("output", {})["file"] = args.output

    run(config)


if __name__ == "__main__":
    main()
