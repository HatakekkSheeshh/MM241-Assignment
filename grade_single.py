import argparse

from grade_all import grade_one_group

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_folder", type=str, required=True)
    args = parser.parse_args()

    grade_one_group(args.group_folder, force=True)
