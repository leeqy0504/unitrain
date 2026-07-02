#!/usr/bin/env python3
"""Unified prediction/inference entry point."""

import argparse

from unitrain import get_runner, load_config


def main():
    parser = argparse.ArgumentParser(description="Unified DL Prediction")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML file")
    parser.add_argument("--source", "-s", required=True, help="Image/video/directory to run inference on")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f">>> Predicting with {config.framework} / {config.model}")

    runner = get_runner(config.framework)
    results = runner.predict(config.to_dict(), args.source)

    print(">>> Prediction complete!")
    return results


if __name__ == "__main__":
    main()
