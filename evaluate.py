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
    AddSub,
    GSM8K,
    AQuA,
)
from solver import PSCoTSolver, ZSCoTSolver

T = TypeVar("T", bound=Problem)
M = TypeVar("M", bound=MultiChoiceProblem)


def number_equal(lhs: str, rhs: str) -> bool:
    """
    Compare two number in str.
    """
    eps = 1e-4
    try:
        return abs(float(lhs) - float(rhs)) < eps
    except ValueError:
        return False


def evaluate_numerical(
    file_path: str, model: Type[T], length_limit: int = None, model_name: str = None
):
    """
    Evaluate the accuracy of Dataset who's numerical
    """
    dataset = load_json(file_path, model, length_limit)
    tot_cnt = len(dataset)
    ps_solver = PSCoTSolver(model_name=model_name)
    ps_correct_cnt = 0
    zs_solver = ZSCoTSolver(model_name=model_name)
    zs_correct_cnt = 0
    for index, numerical in enumerate(dataset):
        logger.info(f"Running on {index + 1} case... total {tot_cnt}")
        # test ps_solver
        ps_solver.set_problem(numerical.problem())
        ps_answer = ps_solver.solve_numerical()
        if number_equal(ps_answer, numerical.answer()):
            ps_correct_cnt += 1
        else:
            logger.warning(
                f"PS failed {index + 1}!!! expected {numerical.answer()}, actual {ps_answer}."
            )
        # test zs_solver
        zs_solver.set_problem(numerical.problem())
        zs_answer = zs_solver.solve_numerical()
        if number_equal(zs_answer, numerical.answer()):
            zs_correct_cnt += 1
        else:
            logger.warning(
                f"ZS failed {index + 1}!!! expected {numerical.answer()}, actual {zs_answer}."
            )
        logger.info(
            f"On {index + 1} case, PS correct {ps_correct_cnt} and ZS correct {zs_correct_cnt}."
        )

    logger.info(f"PS Solver accuracy: {ps_correct_cnt / tot_cnt}")
    logger.info(f"ZS Solver accuracy: {zs_correct_cnt / tot_cnt}")


def evaluate_multichoice(file_path: str, model: Type[T], length_limit: int = None):
    """
    Evaluate the accuracy of Dataset who's multi-choice
    """
    dataset = load_jsonl(file_path, model, length_limit)
    tot_cnt = len(dataset)
    ps_solver = PSCoTSolver()
    ps_correct_cnt = 0
    zs_solver = ZSCoTSolver()
    zs_correct_cnt = 0
    for index, multi_choice in enumerate(dataset):
        logger.info(f"Running on {index + 1} case... total {tot_cnt}")
        # test ps_solver
        ps_solver.set_problem(multi_choice.problem())
        ps_answer = ps_solver.solve_multichoice(multi_choice.options())
        if ps_answer.lower() == multi_choice.answer().lower():
            ps_correct_cnt += 1
        else:
            logger.warning(
                f"PS failed {index + 1}!!! expected {multi_choice.answer()}, actual {ps_answer}."
            )
        # test zs_solver
        zs_solver.set_problem(multi_choice.problem())
        zs_answer = zs_solver.solve_multichoice(multi_choice.options())
        if zs_answer.lower() == multi_choice.answer().lower():
            zs_correct_cnt += 1
        else:
            logger.warning(
                f"ZS failed {index + 1}!!! expected {multi_choice.answer()}, actual {zs_answer}."
            )
        logger.info(
            f"On {index + 1} case, PS correct {ps_correct_cnt} and ZS correct {zs_correct_cnt}."
        )
    logger.info(f"PS Solver accuracy: {ps_correct_cnt / tot_cnt}")
    logger.info(f"ZS Solver accuracy: {zs_correct_cnt / tot_cnt}")


def evaluate_add_sub(file_path: str, model_name: str = None):
    """Evaluate AddSub Dataset"""
    evaluate_numerical(file_path, AddSub, model_name=model_name)


def evaluate_gsm8k(file_path: str, model_name: str = None):
    """Evaluate GSM8K Dataset"""
    evaluate_numerical(file_path, GSM8K, 300, model_name)


def evaluate_aqua(file_path: str):
    """Evaluate AQuA Dataset"""
    evaluate_multichoice(file_path, AQuA)


if __name__ == "__main__":
    logger.add("add_sub-llama3.log", level="INFO")
    evaluate_gsm8k("./dataset/AddSub.json", model_name="llama3:8b")
