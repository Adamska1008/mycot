"""
Load the dataset
"""

import json
from typing import TypeVar, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError, Field


class Problem(ABC, BaseModel):
    """
    Interface of a problem BaseModel
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


# TODO: Define selective problem


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
        try:
            data = json.load(file)
            if isinstance(data, list):
                for item_data in data:
                    try:
                        item = model(**item_data)
                        data_list.append(item)
                    except ValidationError as e:
                        print(f"Validation error for item {item_data}: {e}")
            else:
                print("JSON root element is not an array")

        except json.JSONDecodeError as e:
            print(f"Error reading json file: {e}")

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


class GSM8K(Problem):
    """
    model of a problem from GSM8K dataset
    """

    question: str
    _answer: str = Field(alias="answer")

    def problem(self) -> str:
        return self.question

    def answer(self) -> str:
        return self._answer
