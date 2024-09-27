"""
Doing evaluate stuff
"""

from typing import Type, TypeVar
from loguru import logger
from loader import (
    load_json,
    load_jsonl,
    Problem,
    MultiChoiceProblem,
)
from solver import CoTSolver

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
        if dataset.file_format() == "json"
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
