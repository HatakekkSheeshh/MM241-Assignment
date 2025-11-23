import argparse
import importlib
import json

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError

import gym_cutting_stock
import gymnasium as gym

from configs import CONFIGS
from tqdm import tqdm


def import_submodule(full_name):
    module_name = full_name.split(".")
    submodule_name = module_name[-1]
    module_name = ".".join(module_name[:-1])
    module = importlib.import_module(module_name)
    submodule = getattr(module, submodule_name)
    return submodule


def create_env(config):
    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        **config,
    )
    return env


def run_one_episode(config, policy):
    env = create_env(config)
    observation, info = env.reset(seed=config["seed"])

    terminated = False
    truncated = False

    try:
        while not terminated and not truncated:
            action = policy.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
    except Exception as e:
        print(f"Error: {e}")
        info = {"filled_ratio": 1.0, "trim_loss": 1.0}

    return info


def grade_one_group(group_folder, force=False):
    module_name = "policy" + group_folder[1:]
    module_name = (
        f"student_submissions.{group_folder}.{module_name}.Policy{group_folder[1:]}"
    )
    policy_class = import_submodule(module_name)

    for pid in [1, 2]:
        if not force and os.path.exists(
            f"student_submissions/{group_folder}/grade_p{pid}.json"
        ):
            print(f"{group_folder} is already graded!")
            continue

        results = []
        for config in CONFIGS:
            try:
                policy = policy_class(policy_id=pid)
            except Exception as e:
                print(f"Error: {e}")
                results.append({"filled_ratio": 1.0, "trim_loss": 1.0})
                continue

            executor = ThreadPoolExecutor(max_workers=1)
            # Pass the arguments to the function via `submit`
            future = executor.submit(run_one_episode, config, policy)

            try:
                # Timeout after 900 seconds
                result = future.result(timeout=900)
            except TimeoutError:
                print("Function execution exceeded the time limit!")
                result = {"filled_ratio": 1.0, "trim_loss": 1.0}

            results.append(result)

        # Save the results
        with open(f"student_submissions/{group_folder}/grade_p{pid}.json", "w") as f:
            json.dump(results, f, indent=4)

    return True


def grade_all_groups(args):
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

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(
            tqdm(executor.map(grade_one_group, group_folders),
                 total=len(group_folders))
        )

    if sum(results) == len(group_folders):
        print("Grading completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    grade_all_groups(args)
