"""
Main entry for the project
"""

import threading
import argparse
import itertools
from typing import Type
from evaluate import evaluate_dataset
from solver import ZSCoTSolver, PSCoTSolver, GiveAListSolver, CoTSolver
from loader import AddSub, GSM8K, AQuA, CoinFlip, Problem
from logger import ThreadLogger

logger = ThreadLogger()


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
    dataset_names = ["AddSub", "GSM8K", "AQuA", "CoinFlip"]
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
        "--model",
        type=str,
        help="Model to be tested",
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
    solver_class_map: dict[str, Type[CoTSolver]] = {
        "zero_shot": ZSCoTSolver,
        "plan_and_solve": PSCoTSolver,
        "give_a_list": GiveAListSolver,
    }

    solvers = args.solver
    datasets = args.dataset
    group = itertools.product(solvers, datasets)  # cartesian product
    threads = []
    range_arg = args.range
    model = args.model if args.model else "gpt-4o-mini"

    for solver, dataset in group:
        logger_file = f"./logs/{solver}_{dataset}.log"

        if range_arg is None and dataset == "GSM8K":
            range_arg = range(0, 400)

        solver_cls = solver_class_map[solver]
        dataset_cls: Type[Problem] = globals()[dataset]
        file_path = f"./dataset/{dataset}.{dataset_cls.file_format()}"

        evaluation_thread = threading.Thread(
            target=evaluate_dataset,
            kwargs={
                "file_path": file_path,
                "dataset": dataset_cls,
                "solver": solver_cls,
                "range_arg": range_arg,
                "answer_type": dataset_cls.answer_type(),
                "model_name": model,
            },
        )

        threads.append(evaluation_thread)
        evaluation_thread.start()
        logger.bind(
            evaluation_thread.ident,
            logger_file,
            "DEBUG" if args.debug else "INFO",
        )
        print(f"Starting evaluation for {solver} on {dataset}")

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
