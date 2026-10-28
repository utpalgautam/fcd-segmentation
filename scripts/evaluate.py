import argparse
def main():
    parser = argparse.ArgumentParser(description="Future evaluation entry point")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.parse_args()
    parser.error("Evaluation is not implemented yet. See docs/experiments.md.")
if __name__ == "__main__":
    main()
