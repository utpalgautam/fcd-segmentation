import argparse
def main():
    parser = argparse.ArgumentParser(description="Future training entry point")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.parse_args()
    parser.error("Training is not implemented yet. See docs/experiments.md.")
if __name__ == "__main__":
    main()
