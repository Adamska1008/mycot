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
    length_limit: int | None = None,
    model_name: str | None = None,
    is_numerical: bool = True,
):
    """
    评估数据集的准确性，适用于数值型和多选题

    Args:
        file_path: 数据集文件路径
        dataset: 数据集类型
        solver: 求解器类型
        length_limit: 数据集长度限制
        model_name: 模型名称
        is_numerical: 是否为数值型数据集
    """
    dataset = (
        load_json(file_path, dataset, length_limit)
        if is_numerical
        else load_jsonl(file_path, dataset, length_limit)
    )
    tot_cnt = len(dataset)
    cot_solver = solver(model_name=model_name)
    correct_cnt = 0

    for index, problem in enumerate(dataset):
        logger.info(f"正在运行第 {index + 1} 个案例... 共 {tot_cnt} 个")

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
                f"求解失败 {index + 1}!!! 预期 {problem.answer()}, 实际 {answer}."
            )

        logger.info(f"在第 {index + 1} 个案例中，正确 {correct_cnt} 个。")

    logger.info(f"{solver.name()} 求解器准确率: {correct_cnt / tot_cnt}")


def build_args() -> argparse.Namespace:
    """
    Build args for evaluate
    """
    solver_names = ["zero_shot", "plan_and_solve", "give_a_list"]
    dataset_names = ["AddSub", "GSM8K", "AQuA"]
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
    parser.add_argument(
        "--index",
        type=int,
        help="需要测试的题目序号",
    )
    return parser.parse_args()


if __name__ == "__main__":
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

    length_limit = 400 if args.dataset == "GSM8K" else None

    evaluate_dataset(
        file_path=file_path,
        dataset=dataset_class_map[args.dataset],
        solver=solver_class_map[args.solver],
        length_limit=length_limit,
        is_numerical=is_numerical,
    )
