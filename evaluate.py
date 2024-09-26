"""
Doing evaluate stuff
"""

from typing import Type, TypeVar
import argparse
from loguru import logger
from loader import (
    load_json,
    load_jsonl,
    Problem,
    MultiChoiceProblem,
    AddSub,
    GSM8K,
    AQuA,
)
from solver import CoTSolver, PSCoTSolver, ZSCoTSolver, GiveAListSolver

T = TypeVar("T", bound=Problem)
M = TypeVar("M", bound=MultiChoiceProblem)
S = TypeVar("S", bound=CoTSolver)


def number_equal(lhs: str, rhs: str) -> bool:
    """
    Compare two number in str.
    """
    eps = 1e-4
    try:
        return abs(float(lhs) - float(rhs)) < eps
    except ValueError:
        return False


def evaluate_dataset(
    file_path: str,
    dataset: Type[T],
    solver: Type[S],
    range_arg: range | None = None,
    model_name: str | None = None,
    is_numerical: bool = True,
):
    """
    Evaluate the accuracy of the dataset, applicable to numerical and multiple-choice questions

    Args:
        file_path: Dataset file path
        dataset: Dataset type
        solver: Solver type
        range_arg: The range of the problems to be loaded
        model_name: Model name
        is_numerical: Whether it's a numerical dataset
    """
    dataset = (
        load_json(file_path, dataset, range_arg=range_arg)
        if is_numerical
        else load_jsonl(file_path, dataset, range_arg=range_arg)
    )
    tot_cnt = len(dataset)
    cot_solver = solver(model_name=model_name)
    correct_cnt = 0

    for index, problem in enumerate(dataset):
        logger.info(f"Running case {index + 1}... Total {tot_cnt}")

        cot_solver.set_problem(problem.problem())
        answer = (
            cot_solver.solve_numerical()
            if is_numerical
            else cot_solver.solve_multichoice(problem.options())
        )
        if (is_numerical and number_equal(answer, problem.answer())) or (
            not is_numerical and answer.lower() == problem.answer().lower()
        ):
            correct_cnt += 1
        else:
            logger.warning(
                f"Solving failed {index + 1}!!! Expected {problem.answer()}, Got {answer}."
            )

        logger.info(f"In case {index + 1}, correct {correct_cnt}.")

    logger.info(f"{solver.__name__} solver accuracy: {correct_cnt / tot_cnt}")


####################################
# Below are logic for command line #
####################################


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
