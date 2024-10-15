"""
Doing evaluate stuff
"""

import itertools
import threading
from typing import Type, TypeVar
from loader import load_json, load_jsonl, Problem, MultiChoiceProblem, AnswerType, GSM8K
from solver import CoTSolver
from logger import ThreadLogger

logger = ThreadLogger()


P = TypeVar("P", bound=Problem)
M = TypeVar("M", bound=MultiChoiceProblem)
S = TypeVar("S", bound=CoTSolver)


def answer_equal(answer: str, output: str, answer_type: AnswerType) -> bool:
    """
    Check if the answer is equal to the output
    """

    def num_equal(lhs: str, rhs: str) -> bool:
        eps = 1e-4
        try:
            return abs(float(lhs) - float(rhs)) < eps
        except ValueError:
            return False

    def option_equal(lhs: str, rhs: str) -> bool:
        return lhs.lower() == rhs.lower()

    def boolean_equal(lhs: str, rhs: str) -> bool:
        """
        Assuming answer is yes or no. May change in the future.
        """
        return lhs.strip(".").lower() == rhs.strip(".").lower()

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
    dataset: Type[P],
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
            output = cot_solver.solve_boolean(["yes", "no"])
        else:
            assert False
        answer = problem.answer()
        if answer_equal(answer, output, answer_type):
            correct_cnt += 1
        else:
            logger.warning(
                f"Solving failed {index + 1}!!! Expected {answer}, Got {output}."
            )
            cot_solver.agent.debug()

        logger.info(f"In case {index + 1}, correct {correct_cnt}.")

    logger.info(f"{solver.__name__} solver accuracy: {correct_cnt / tot_cnt}")


def evaluate_in_threads(
    solvers: list[Type[S]],
    datasets: list[Type[P]],
    range_arg: range | None = None,
    model: str = "gpt-4o-mini",
    debug: bool = False,
):
    """
    Evaluate datasets and solvers simultaneously.
    """
    group = itertools.product(solvers, datasets)
    threads = []
    for solver, dataset in group:
        log_file = f"./logs/{solver.__name__}_{dataset.__name__}.log"
        if range_arg is None and dataset is GSM8K:
            range_arg = range(0, 400)
        dataset_path = f"./dataset/{dataset.__name__}.{dataset.file_format()}"
        evaluation_thread = threading.Thread(
            target=evaluate_dataset,
            kwargs={
                "file_path": dataset_path,
                "dataset": dataset,
                "solver": solver,
                "range_arg": range_arg,
                "answer_type": dataset.answer_type(),
                "model_name": model,
            },
        )

        threads.append(evaluation_thread)
        evaluation_thread.start()
        logger.bind(
            evaluation_thread.ident,
            log_file,
            "DEBUG" if debug else "INFO",
        )
        print(f"Starting evaluation for {solver} on {dataset}")
    
    for thread in threads:
        thread.join()
