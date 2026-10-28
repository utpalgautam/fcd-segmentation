import argparse
def main():
    parser = argparse.ArgumentParser(description="Future prediction entry point")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.parse_args()
    parser.error("Inference is not implemented yet. See docs/experiments.md.")
if __name__ == "__main__":
    main()
