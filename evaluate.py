"""
Doing evaluate stuff
"""

from typing import Type, TypeVar
from loguru import logger
from loader import load_json, Problem, AddSub, GSM8K
from solver import PSCoTSolver, ZSCoTSolver

T = TypeVar("T", bound=Problem)


def number_equal(lhs: str, rhs: str) -> bool:
    """
    Compare two number in str.
    """
    eps = 1e-4
    try:
        return abs(float(lhs) - float(rhs)) < eps
    except ValueError:
        return False


def evaluate(file_path: str, model: Type[T], length_limit: int = None):
    """
    Evaluate the accuracy of AddSub Dataset
    """
    dataset = load_json(file_path, model, length_limit)
    tot_cnt = len(dataset)
    ps_solver = PSCoTSolver()
    ps_correct_cnt = 0
    zs_solver = ZSCoTSolver()
    zs_correct_cnt = 0
    for index, add_sub in enumerate(dataset):
        logger.info(f"Running on {index + 1} case... total {tot_cnt}")
        # test ps_solver
        ps_solver.set_problem(add_sub.problem())
        ps_answer = ps_solver.solve()
        if number_equal(ps_answer, add_sub.answer()):
            ps_correct_cnt += 1
        else:
            logger.warning(
                f"PS failed {index + 1}!!! expected {add_sub.answer()}, actual {ps_answer}."
            )
        # test zs_solver
        zs_solver.set_problem(add_sub.problem())
        zs_answer = zs_solver.solve()
        if number_equal(zs_answer, add_sub.answer()):
            zs_correct_cnt += 1
        else:
            logger.warning(
                f"ZS failed {index + 1}!!! expected {add_sub.answer()}, actual {zs_answer}."
            )
        logger.info(
            f"On {index + 1} case, PS correct {ps_correct_cnt} and ZS correct {zs_correct_cnt}."
        )

    logger.info(f"PS Solver accuracy: {ps_correct_cnt / tot_cnt}")
    logger.info(f"ZS Solver accuracy: {zs_correct_cnt / tot_cnt}")


def evaluate_add_sub(file_path: str):
    """Evaluate AddSub Dataset"""
    evaluate(file_path, AddSub)


def evaluate_gsm8k(file_path: str):
    """Evaluate GSM8K Dataset"""
    evaluate(file_path, GSM8K, 300)


if __name__ == "__main__":
    logger.add("output.log", level="INFO")
    evaluate_gsm8k("./dataset/gsm8k.json")
