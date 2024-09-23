"""
Load the dataset
"""

import json
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


def load_json(file_path: str, model: Type[T]) -> list[T]:
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

    return data_list


class AddSub(BaseModel):
    """
    model of a problem from AddSub dataset
    """

    iIndex: int
    lEquations: list[str]
    lSolutions: list[str]
    sQuestion: str

    def problem(self) -> str:
        """
        Give the description of the problem
        """
        return self.sQuestion

    def answer(self) -> str:
        """
        Give the answer of the problem
        """
        return self.lSolutions[0]
