"""
Codes related to CoTSolver
"""

from abc import ABC, abstractmethod
from agent import OpenAIChatAgent

COT_AI_PROMPT = (
    "Let's first understand the problem, extract relevant variables and their corresponding numerals,"
    "and **devise a plan**. Then, let's **carry out the plan**, caculate intermediate results"
    "(pay attention to calculation and common sense), solve the problem step by step, and show the answer."
)


# def extract_numerical_answer(solve_fn):
#     """
#     Enhance the output to give a single number
#     """

#     def wrapper(self, *args, **kwargs) -> str:
#         solve_fn(self, *args, **kwargs)
#         assert self.agent is not None, "use extract wrapper on a class without agent"
#         return self.agent.post_human(
#             "Therefore the answer is?"
#             "Output only a real number(e.g., 3.14) and do not use fractional form(like 1/2 or 3/4)."
#             "If your answer is a repeating decimal, round it to six decimal places."
#         )

#     return wrapper


# def extract_multiple_choice_answer(
#     solve_fn, option_a: str, option_b: str, option_c: str, option_d: str
# ):
#     """
#     Enhance the output to choose from 4 option
#     """

#     def wrapper(self, *args, **kwargs) -> str:
#         solve_fn(self, args, kwargs)
#         assert self.agent is not None, "use extract wrapper on a class without agent"
#         return self.agent.post_human(
#             "Here are four options for the answer:\n"
#             f"A: {option_a}\n"
#             f"B: {option_b}\n"
#             f"C: {option_c}\n"
#             f"D: {option_d}\n"
#             "Please choose and output one of the following: A, B, C or D."
#         )

#     return wrapper


class CoTSolver(ABC):
    """
    Chain-of-Thought Solver
    """

    @abstractmethod
    def solve(self) -> str:
        """
        Solve the problem.

        Returns:
            - the final answer
        """

    @property
    @abstractmethod
    def agent(self):
        """
        get the agent
        """

    def solve_numerical(self) -> str:
        """
        Solve a numerical problem, e.g. return only a number.
        """
        self.solve()
        return self.agent.post_human(
            "Therefore the answer is?"
            "Output only a real number(e.g., 3.14) and do not use fractional form(like 1/2 or 3/4)."
            "If your answer is a repeating decimal, round it to six decimal places."
        )

    def solve_multichoice(
        self, option_a: str, option_b: str, option_c: str, option_d: str
    ) -> str:
        """
        Solve a multiple-choice question, e.g. return one of [A, B, C, D]
        """
        self.solve()
        return self.agent.post_human(
            "Here are four options for the answer:\n"
            f"A: {option_a}\n"
            f"B: {option_b}\n"
            f"C: {option_c}\n"
            f"D: {option_d}\n"
            "Please choose and output one of the following: A, B, C or D."
        )


class ZSCoTSolver(CoTSolver):
    """
    Wraps an agent and use zero-shot CoT to solve a problem(usually math).
    """

    def __init__(self, problem: str = None):
        self._problem = problem
        self._agent = OpenAIChatAgent()

    @property
    def agent(self):
        return self._agent

    def set_problem(self, probelm: str) -> None:
        """
        Simple Setter
        """
        self._problem = probelm

    def solve(self) -> str:
        """
        Solve the problem.

        Returns:
            - a number indicates the final answer
        """
        self.agent.clear_history()
        self.agent.store_human(self._problem)
        return self.agent.post_ai("Let's think step by step.")


class PSCoTSolver(CoTSolver):
    """
    Wraps an agent and use Plan-and-Solve CoT to solve a problem(usually math).
    """

    def __init__(self, problem: str = None):
        self._problem = problem
        self._agent = OpenAIChatAgent()

    @property
    def agent(self):
        return self._agent

    def set_problem(self, problem: str):
        """
        Simple Setter
        """
        self._problem = problem

    def solve(self) -> str:
        """
        Solve the problem.

        Returns:
            - a number indicates the final answer
        """
        self.agent.clear_history()
        self.agent.store_human(self._problem)
        return self.agent.post_ai(COT_AI_PROMPT)


if __name__ == "__main__":
    solver = ZSCoTSolver(
        problem=(
            "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Graces weighs."
            "What are theire combined weights in pounds?"
        )
    )

    print(solver.solve_numerical())
