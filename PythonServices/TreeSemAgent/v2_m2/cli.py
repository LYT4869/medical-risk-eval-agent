"""M2 synthetic execution CLI. Explicit M1 Contract confirmation is prerequisite."""
import argparse
import json

from .service import M2SyntheticService


def main():
    parser = argparse.ArgumentParser(description="M2 P001 synthetic level-0")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("run_id", help="confirmed run_id from M1 CLI")
    args = parser.parse_args()
    result = M2SyntheticService(args.data_dir).execute(args.run_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
