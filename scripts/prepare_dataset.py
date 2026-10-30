"""Preprocessing entry point reserved until protocol is approved."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Future preprocessing pipeline")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    parser.parse_args()
    parser.error("Not implemented yet. See docs/experiments.md.")


if __name__ == "__main__":
    main()
