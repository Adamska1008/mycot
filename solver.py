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

    def solve_multichoice(self, options: dict[str, str]) -> str:
        """
        Solve a multiple-choice question, e.g. return one of [A, B, C, D, E, ...]
        """
        self.solve()
        options_lines = []
        for k, v in options.items():
            options_lines.append(f"{k}: {v}")
        return self.agent.post_human(
            f"Here are {len(options_lines)} options for the answer:\n"
            + "\n".join(options_lines)
            + "\n"
            + "Please choose and output one of the upper letter of the options, e.g. A"
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
