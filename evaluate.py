"""
Doing evaluate stuff
"""

from enum import Enum
from typing import Type, TypeVar
from loguru import logger
from loader import (
    load_json,
    load_jsonl,
    Problem,
    MultiChoiceProblem,
    AnswerType
)
from solver import CoTSolver

T = TypeVar("T", bound=Problem)
M = TypeVar("M", bound=MultiChoiceProblem)
S = TypeVar("S", bound=CoTSolver)


def answer_equal(answer: str, output: str, answer_type: AnswerType) -> bool:
    def num_equal(lhs: str, rhs: str) -> bool:
        eps = 1e-4
        try:
            return abs(float(lhs, float(rhs))) < eps
        except ValueError:
            return False

    def option_equal(lhs: str, rhs: str) -> bool:
        return lhs.lower() == rhs.lower()
    
    def boolean_equal(lhs: str, rhs: str) -> bool:
        """
        Assuming answer is yes or no. May change in the future.
        """
        return lhs.lower() == rhs.lower()

    if answer_type is AnswerType.Number:
        return num_equal(answer, output)
    elif answer_type is AnswerType.Option:
        return option_equal(answer, output)
    elif answer_type is AnswerType.Boolean:
        return boolean_equal(answer, output)
    else:
        assert False


def evaluate_dataset(
    file_path: str,
    dataset: Type[T],
    solver: Type[S],
    answer_type: AnswerType,
    range_arg: range | None = None,
    model_name: str | None = None,
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
        if answer_type is AnswerType.Number:
            output = cot_solver.solve_numerical()
        elif answer_type is AnswerType.Option:
            output = cot_solver.solve_multichoice(problem.options())
        elif answer_type is AnswerType.Boolean:
            output = cot_solver.solve_boolean()
        else:
            assert False
        answer = problem.answer()
        if answer_equal(answer, output, answer_type):
            correct_cnt += 1
        else:
            logger.warning(
                f"Solving failed {index + 1}!!! Expected {problem.answer()}, Got {answer}."
            )

        logger.info(f"In case {index + 1}, correct {correct_cnt}.")

    logger.info(f"{solver.__name__} solver accuracy: {correct_cnt / tot_cnt}")
