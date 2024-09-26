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
from solver import PSCoTSolver, ZSCoTSolver, GiveAListSolver

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


def evaluate_problem(
    file_path: str,
    dataset: Type[T],
    length_limit: int = None,
    model_name: str = None,
    is_numerical: bool = True,
):
    """
    评估数据集的准确性，适用于数值型和多选题
    """
    dataset = (
        load_json(file_path, dataset, length_limit)
        if is_numerical
        else load_jsonl(file_path, dataset, length_limit)
    )
    tot_cnt = len(dataset)
    ps_solver = PSCoTSolver(model_name=model_name)
    ps_correct_cnt = 0
    zs_solver = ZSCoTSolver(model_name=model_name)
    zs_correct_cnt = 0

    for index, problem in enumerate(dataset):
        logger.info(f"正在运行第 {index + 1} 个案例... 共 {tot_cnt} 个")

        # 测试 ps_solver
        ps_solver.set_problem(problem.problem())
        ps_answer = (
            ps_solver.solve_numerical()
            if is_numerical
            else ps_solver.solve_multichoice(problem.options())
        )
        if (is_numerical and number_equal(ps_answer, problem.answer())) or (
            not is_numerical and ps_answer.lower() == problem.answer().lower()
        ):
            ps_correct_cnt += 1
        else:
            logger.warning(
                f"PS 失败 {index + 1}!!! 预期 {problem.answer()}, 实际 {ps_answer}."
            )

        # 测试 zs_solver
        zs_solver.set_problem(problem.problem())
        zs_answer = (
            zs_solver.solve_numerical()
            if is_numerical
            else zs_solver.solve_multichoice(problem.options())
        )
        if (is_numerical and number_equal(zs_answer, problem.answer())) or (
            not is_numerical and zs_answer.lower() == problem.answer().lower()
        ):
            zs_correct_cnt += 1
        else:
            logger.warning(
                f"ZS 失败 {index + 1}!!! 预期 {problem.answer()}, 实际 {zs_answer}."
            )

        logger.info(
            f"在第 {index + 1} 个案例中，PS 正确 {ps_correct_cnt} 个，ZS 正确 {zs_correct_cnt} 个。"
        )

    logger.info(f"PS 求解器准确率: {ps_correct_cnt / tot_cnt}")
    logger.info(f"ZS 求解器准确率: {zs_correct_cnt / tot_cnt}")


def evaluate_multichoice(
    input_file_path: str,
    model: Type[T],
    length_limit: int = None,
    model_name: str = None,
):
    """
    Evaluate the accuracy of Dataset who's multi-choice
    """
    dataset = load_jsonl(input_file_path, model, length_limit)
    tot_cnt = len(dataset)
    ps_solver = PSCoTSolver(model_name=model_name)
    ps_correct_cnt = 0
    zs_solver = ZSCoTSolver(model_name=model_name)
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


# Below are main logic


def build_args() -> argparse.Namespace:
    """
    Build args for evaluate
    """
    solver_names = [cls.name() for cls in [ZSCoTSolver, PSCoTSolver, GiveAListSolver]]
    dataset_names = [cls.name() for cls in [AQuA, AddSub, GSM8K]]
    parser = argparse.ArgumentParser(
        description="使用这个脚本来快速测试某个Solver解决数据集中某一个题目的效果"
    )
    parser.add_argument(
        "--solver",
        type=str,
        help="需要测试的Solver",
        required=True,
        choices=solver_names,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="需要测试的数据集",
        required=True,
        choices=dataset_names,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    dataset_class_map = {
        "AddSub": AddSub,
        "GSM8K": GSM8K,
        "AQuA": AQuA,
    }
    logger_file = f"./logs/{args.solver}_{args.dataset}.log"
    logger.remove()
    logger.add(logger_file)

    file_path = f"./dataset/{args.dataset}.{'json' if args.dataset in ['AddSub', 'GSM8K'] else 'jsonl'}"
    is_numerical = args.dataset in ["AddSub", "GSM8K"]

    evaluate_problem(
        file_path=file_path,
        dataset=dataset_class_map[args.dataset],
        is_numerical=is_numerical,
    )
