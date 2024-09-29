"""
Codes related to CoTSolver
"""

from abc import ABC, abstractmethod
from agent import ChatAgent

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

    @abstractmethod
    def set_problem(self, str):
        """
        Set the problem description to be solved.
        NOTICE: The options should not be contained if it's a multiple choices problem.
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
            "Output only a real number(e.g., 3.14). Do not use fractional form(like 1/2 or 3/4)."
            "Do not show a equation like 1 + 1 = 2. In this case, output 2 only."
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
            + "Please choose and output one of the upper letter of the options, e.g. A. "
            "If you think none of them is correct, just output the most likely letter."
            "Do not output anything beside the letter like A"
            "Please do not use bold formatting in your output."
        )

    def solve_boolean(self, boolean_expression: tuple[str, str]):
        """
        Args:
            boolean_expression: two str in a tuple, presenting true and false. e.g. ["True", "False"]
        """
        self.solve()
        return self.post_human(
            "Decide the answer to the problem to be true or false."
            f"If you think it's true, output {boolean_expression[0]},"
            f"else output {boolean_expression[1]}"
        )

class ZSCoTSolver(CoTSolver):
    """
    Wraps an agent and use zero-shot CoT to solve a problem(usually math).
    """

    def __init__(self, problem: str = None, model_name: str = None):
        self._problem = problem
        self._agent = ChatAgent(model_name=model_name)

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

    def __init__(self, problem: str = None, model_name: str = None):
        self._problem = problem
        self._agent = ChatAgent(model_name=model_name)

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


class GiveAListSolver(CoTSolver):
    """
    Prompot the agent to give a list of steps to solve the problem
    """

    def __init__(self, problem: str = None, model_name: str = None):
        self._problem = problem
        self._agent = ChatAgent(
            system_prompt=(
                "You should give a list of steps to solve the problem, from 1 to N steps."
                "Each step should start with a number and a dot, like '1. '."
                "The steps should be as detailed as possible, and each step should be a complete sentence."
                "The last step should be the answer."
            ),
            model_name=model_name,
        )

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
        Prompt the agent to give a list of steps to solve the problem
        """
        self.agent.clear_history()
        self.agent.store_human(self._problem)
        return self.agent.post_ai(
            "Ok, I will think step by step and give you a list of steps to solve the problem.\n"
            "1. "
        )
