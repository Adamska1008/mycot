"""
Codes related to CoTSolver
"""

from agent import OpenAIChatAgent

COT_AI_PROMPT = (
    "Let's first understand the problem, extract relevant variables and their corresponding numerals,"
    "and devise a plan. Then, let's carry out the plan, caculate intermediate results"
    "(pay attention to calculation and common sense), solve the problem step by step, and show the answer."
)


def extract_numerical_answer(solve_fn):
    """
    Enhance the output
    """

    def wrapper(self, *args, **kwargs):
        solve_fn(self, *args, **kwargs)
        assert self.agent is not None, "use extract wrapper on a class without agent"
        return self.agent.post_human(
            "Therefore the answer is? "
            "output only a real number(e.g., 3.14) and do not use fractional form(like 1/2 or 3/4)."
            "If your answer is a repeating decimal, round it to six decimal places."
        )

    return wrapper


class ZSCoTSolver:
    """
    Wraps an agent and use zero-shot CoT to solve a problem(usually math).
    """

    def __init__(self, problem: str = None):
        self.problem = problem
        self.agent = OpenAIChatAgent()

    def set_problem(self, probelm: str) -> None:
        """
        Simple Setter
        """
        self.problem = probelm

    @extract_numerical_answer
    def solve(self) -> str:
        """
        Solve the problem.

        Returns:
            - a number indicates the final answer
        """
        self.agent.clear_history()
        self.agent.store_human(self.problem)
        return self.agent.post_ai("Let's think step by step.")


class PSCoTSolver:
    """
    Wraps an agent and use Plan-and-Solve CoT to solve a problem(usually math).
    """

    def __init__(self, problem: str = None):
        self.problem = problem
        self.agent = OpenAIChatAgent()

    def set_problem(self, problem: str):
        """
        Simple Setter
        """
        self.problem = problem

    @extract_numerical_answer
    def solve(self) -> str:
        """
        Solve the problem.

        Returns:
            - a number indicates the final answer
        """
        self.agent.clear_history()
        self.agent.store_human(self.problem)
        return self.agent.post_ai(COT_AI_PROMPT)


if __name__ == "__main__":
    solver = ZSCoTSolver(
        problem=(
            "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Graces weighs."
            "What are theire combined weights in pounds?"
        )
    )

    print(solver.solve())
