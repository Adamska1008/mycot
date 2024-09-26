"""
Main entry for the project
"""

import argparse
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
        type=str,
        help="Solver to be tested",
        required=True,
        choices=solver_names,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to be tested",
        required=True,
        choices=dataset_names,
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
    logger_file = f"./logs/{args.solver}_{args.dataset}.log"
    logger.remove()
    logger.add(logger_file)

    file_path = f"./dataset/{args.dataset}.{'json' if args.dataset in ['AddSub', 'GSM8K'] else 'jsonl'}"
    is_numerical = args.dataset in ["AddSub", "GSM8K"]

    range_arg = args.range

    if range_arg is None and args.dataset == "GSM8K":
        range_arg = range(0, 400)

    evaluate_dataset(
        file_path=file_path,
        dataset=dataset_class_map[args.dataset],
        solver=solver_class_map[args.solver],
        range_arg=range_arg,
        is_numerical=is_numerical,
    )


if __name__ == "__main__":
    main()
