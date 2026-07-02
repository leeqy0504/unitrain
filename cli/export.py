#!/usr/bin/env python3
"""Unified model export entry point."""

import argparse

from unitrain import get_runner, load_config


def main():
    parser = argparse.ArgumentParser(description="Unified DL Model Export")
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML file")
    parser.add_argument("--format", "-f", default=None, help="Export format (onnx, tensorrt, etc.)")
    args = parser.parse_args()

    config = load_config(args.config)
    export_format = args.format or config.export_format
    print(f">>> Exporting {config.framework} / {config.model} to {export_format}")

    runner = get_runner(config.framework)
    output_path = runner.export(config.to_dict(), export_format)

    print(f">>> Exported to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
