import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    group_folders = os.listdir("student_submissions")
    group_folders.remove("s2210xxx")

    # Sort the group folders
    group_folders = sorted(group_folders)

    # Ensure that the group_folders are folders
    group_folders = [
        group_folder
        for group_folder in group_folders
        if os.path.isdir(f"student_submissions/{group_folder}")
        and group_folder.startswith("s")
    ]

    score_dict = {
        "group_folder": [],
        "overall_score": [],
        "avg_filled_ratio": [],
        "avg_trim_loss": [],
        "notes": [],
    }

    for pid in [1, 2]:
        for eid in range(10):
            score_dict[f"p{pid}_e{eid}_filled_ratio"] = []
            score_dict[f"p{pid}_e{eid}_trim_loss"] = []

    for group_folder in tqdm(group_folders, desc="Gathering scores"):
        grade_p1_path = f"student_submissions/{group_folder}/grade_p1.json"
        grade_p2_path = f"student_submissions/{group_folder}/grade_p2.json"

        if not os.path.exists(grade_p1_path) or not os.path.exists(grade_p2_path):
            print(f"{group_folder} failed!")
            score_dict["group_folder"].append(group_folder)
            score_dict["avg_filled_ratio"].append(1.0)
            score_dict["avg_trim_loss"].append(1.0)
            score_dict["notes"].append("Failed to grade!")
            for pid in [1, 2]:
                for eid in range(10):
                    score_dict[f"p{pid}_e{eid}_filled_ratio"].append(1.0)
                    score_dict[f"p{pid}_e{eid}_trim_loss"].append(1.0)
            continue

        with open(grade_p1_path, "r") as f:
            grade_p1 = json.load(f)

        with open(grade_p2_path, "r") as f:
            grade_p2 = json.load(f)

        avg_filled_ratio = 0
        avg_trim_loss = 0

        for eid in range(10):
            p1_eid = grade_p1[eid]
            p2_eid = grade_p2[eid]

            score_dict[f"p1_e{eid}_filled_ratio"].append(
                p1_eid["filled_ratio"])
            score_dict[f"p1_e{eid}_trim_loss"].append(p1_eid["trim_loss"])
            score_dict[f"p2_e{eid}_filled_ratio"].append(
                p2_eid["filled_ratio"])
            score_dict[f"p2_e{eid}_trim_loss"].append(p2_eid["trim_loss"])

            avg_filled_ratio += p1_eid["filled_ratio"] + p2_eid["filled_ratio"]
            avg_trim_loss += p1_eid["trim_loss"] + p2_eid["trim_loss"]

        avg_filled_ratio /= 20
        avg_trim_loss /= 20

        score_dict["group_folder"].append(group_folder)
        score_dict["avg_filled_ratio"].append(avg_filled_ratio)
        score_dict["avg_trim_loss"].append(avg_trim_loss)
        score_dict["notes"].append("")

    # Find the minimum of filled_ratio and trim_loss
    best_filled_ratio = {}
    best_trim_loss = {}
    for pid in [1, 2]:
        for eid in range(10):
            filled_ratio = score_dict[f"p{pid}_e{eid}_filled_ratio"]
            trim_loss = score_dict[f"p{pid}_e{eid}_trim_loss"]

            if eid in best_filled_ratio:
                if np.min(filled_ratio) < best_filled_ratio[eid]:
                    best_filled_ratio[eid] = np.min(filled_ratio)
            else:
                best_filled_ratio[eid] = np.min(filled_ratio)

            if eid in best_trim_loss:
                if np.min(trim_loss) < best_trim_loss[eid]:
                    best_trim_loss[eid] = np.min(trim_loss)
            else:
                best_trim_loss[eid] = np.min(trim_loss)

    best_filled_ratio = np.array([best_filled_ratio[eid] for eid in range(10)])
    best_trim_loss = np.array([best_trim_loss[eid] for eid in range(10)])

    for gid, group_folder in enumerate(tqdm(group_folders, desc="Computing scores")):
        group_filled_ratio = []
        group_trim_loss = []

        for pid in [1, 2]:
            for eid in range(10):
                filled_ratio = score_dict[f"p{pid}_e{eid}_filled_ratio"][gid]
                trim_loss = score_dict[f"p{pid}_e{eid}_trim_loss"][gid]

                group_filled_ratio.append(filled_ratio)
                group_trim_loss.append(trim_loss)

        group_filled_ratio = np.array(group_filled_ratio).reshape(2, -1)
        group_trim_loss = np.array(group_trim_loss).reshape(2, -1)

        group_filled_ratio = np.min(group_filled_ratio, axis=0)
        group_trim_loss = np.min(group_trim_loss, axis=0)

        group_score = 0.1 * np.mean(
            1 + best_filled_ratio - group_filled_ratio
        ) + 0.9 * np.mean(1 + best_trim_loss - group_trim_loss)

        score_dict["overall_score"].append(group_score)

    score_df = pd.DataFrame(score_dict)
    score_df.to_excel("scores.xlsx", index=False)

    score_by_student = {
        "student_id": [],
        "overall_score": [],
    }
    for _, row in score_df.iterrows():
        group_folder = row["group_folder"]
        student_ids = group_folder[1:].split("_")
        for student_id in student_ids:
            score_by_student["student_id"].append(student_id)
            score_by_student["overall_score"].append(row["overall_score"])

    score_by_student_df = pd.DataFrame(score_by_student)
    score_by_student_df.to_excel("scores_by_student.xlsx", index=False)
