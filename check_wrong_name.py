import os
import subprocess
from collections import Counter

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    all_submission_folders = os.listdir("student_submissions")
    # Filter out non-folders
    all_submission_folders = [
        folder
        for folder in all_submission_folders
        if os.path.isdir(f"student_submissions/{folder}")
    ]

    # Remove example folder: s2210xxx
    all_submission_folders.remove("s2210xxx")

    # Sort the folders
    all_submission_folders.sort()

    errors = {"folder": [], "error": []}
    for folder in all_submission_folders:
        list_id = folder[1:]
        files = os.listdir(f"student_submissions/{folder}")
        target_file = f"policy{list_id}.py"

        if target_file not in files:
            print(f"Missing {target_file} in {folder}")
            errors["folder"].append(folder)
            errors["error"].append("Missing file")
        else:
            # Open the file and chech there is a class named Policy{list_id}
            with open(f"student_submissions/{folder}/{target_file}", "r") as f:
                content = f.read()
                if f"class Policy{list_id}(Policy):" not in content:
                    print(f"Wrong class name in {folder}/{target_file}")
                    errors["folder"].append(folder)
                    errors["error"].append("Wrong class name")

    df = pd.DataFrame(errors)
    df.to_csv("wrong_name.csv", index=False)
