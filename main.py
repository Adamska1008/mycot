"""
Main entry for the project
"""

import threading
import argparse
import itertools
from loguru import logger
from evaluate import evaluate_dataset
from solver import ZSCoTSolver, PSCoTSolver, GiveAListSolver
from loader import AddSub, GSM8K, AQuA


def parse_range(range_str: str) -> range:
    """
    Parse string to range object
    """
    try:
        range_parts = range_str.split(",")
        if len(range_parts) == 1:
            # If there's only one number, return range(n, n+1)
            n = int(range_parts[0])
            return range(n, n + 1)
        elif len(range_parts) == 2:
            # If there are two numbers, return range(start, end)
            return range(int(range_parts[0]), int(range_parts[1]))
        else:
            raise ValueError("Invalid input format")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Range must be a single integer or two comma-separated integers"
        ) from exc


def build_args() -> argparse.Namespace:
    """
    Build args
    """
    solver_names = ["zero_shot", "plan_and_solve", "give_a_list"]
    dataset_names = ["AddSub", "GSM8K", "AQuA"]
    parser = argparse.ArgumentParser(
        description="Use this script to quickly test the effect of a Solver solving a problem in the dataset"
    )
    parser.add_argument(
        "--solver",
        nargs="+",
        type=str,
        help="Solver to be tested",
        required=True,
        choices=solver_names,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=str,
        help="Dataset to be tested",
        required=True,
        choices=dataset_names,
    )
    parser.add_argument(
        "--debug", action="store_true", help="set logger to debug level"
    )
    parser.add_argument(
        "--range",
        type=parse_range,
        help="Range of the problems to be tested",
    )
    return parser.parse_args()


def main():
    """
    Main function to avoid global namespace pollution.
    """
    args = build_args()
    dataset_class_map = {
        "AddSub": AddSub,
        "GSM8K": GSM8K,
        "AQuA": AQuA,
    }
    solver_class_map = {
        "zero_shot": ZSCoTSolver,
        "plan_and_solve": PSCoTSolver,
        "give_a_list": GiveAListSolver,
    }

    solvers = args.solver
    datasets = args.dataset
    group = itertools.product(solvers, datasets)  # cartesian product
    logger.remove()
    threads = []
    range_arg = args.range

    for solver, dataset in group:
        logger_file = f"./logs/{solver}_{dataset}.log"
        logger.add(logger_file, level="DEBUG" if args.debug else "INFO")
        file_path = f"./dataset/{dataset}.{'json' if dataset in ['AddSub', 'GSM8K'] else 'jsonl'}"
        is_numerical = dataset in ["AddSub", "GSM8K"]

        if range_arg is None and dataset == "GSM8K":
            range_arg = range(0, 400)

        evaluation_thread = threading.Thread(
            target=evaluate_dataset,
            kwargs={
                "file_path": file_path,
                "dataset": dataset_class_map[dataset],
                "solver": solver_class_map[solver],
                "range_arg": range_arg,
                "is_numerical": is_numerical,
            },
        )
        threads.append(evaluation_thread)
        evaluation_thread.start()
        print(f"Starting evaluation for {solver} on {dataset}")

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
