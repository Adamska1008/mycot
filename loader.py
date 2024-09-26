"""
Load the dataset
"""

import json
from typing import TypeVar, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class Problem(ABC, BaseModel):
    """
    Interface of a problem BaseModel
    """

    @abstractmethod
    def dataset_name() -> str:
        """
        Dataset name of the problem class
        """

    @abstractmethod
    def problem(self) -> str:
        """
        Give the description of the problem
        """

    @abstractmethod
    def answer(self) -> str:
        """
        Give the answer of the problem
        """


class MultiChoiceProblem(Problem):
    """
    Interface of a multiple choice problem
    The return of `answer()` must be one of [A, B, C, D, ...]
    """

    @abstractmethod
    def options(self) -> dict[str, str]:
        """
        Option A of the problem

        Returns:
            A dict containing four keys: [A, B, C, D, ...], and their corresponding values
            Notice that the values should not contain the option(e.g."A") itself
        """


T = TypeVar("T", bound=Problem)


def load_json(file_path: str, model: Type[T], length_limit: int = None) -> list[T]:
    """
    Load a file which is a large json array and return a list of model instances.

    Parameters:
    - file_path: The path to the JSON file.
    - model: The Pydantic model class to use for validation.

    Returns:
    - A list of model instances.
    """
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        for item_data in data:
            item = model(**item_data)
            data_list.append(item)

    if length_limit:
        return data_list[:length_limit]
    return data_list


def load_jsonl(file_path: str, model: Type[T], length_limit: int = None) -> list[T]:
    """Similar"""
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        json_list = list(file)
    for line in json_list:
        item_data = json.loads(line)
        item = model(**item_data)
        data_list.append(item)
    if length_limit:
        return data_list[:length_limit]
    return data_list


class AddSub(Problem):
    """
    model of a problem from AddSub dataset
    """

    iIndex: int
    lEquations: list[str]
    lSolutions: list[str]
    sQuestion: str

    def problem(self) -> str:
        return self.sQuestion

    def answer(self) -> str:
        return self.lSolutions[0]

    def name() -> str:
        return "AddSub" 


class GSM8K(Problem):
    """
    model of a problem from GSM8K dataset
    """

    question: str
    raw_answer: float = Field(alias="answer")

    def problem(self) -> str:
        return self.question

    def answer(self) -> str:
        return str(self.raw_answer)
    
    def name() -> str:
        return "GSM8K"


class AQuA(MultiChoiceProblem):
    """
    model of a problem from AQuA dataset
    """

    question: str
    raw_options: list[str] = Field(alias="options")
    rationale: str
    correct: str

    def problem(self) -> str:
        return self.question

    def answer(self) -> str:
        return self.correct

    def name() -> str:
        return "AQuA"

    def options(self) -> dict[str, str]:
        res = {}
        for index, option in enumerate(self.raw_options):
            letter = chr(index + 65)
            while option.startswith(f"{letter}("):
                option = option[2:]
            res[letter] = option
        return res
