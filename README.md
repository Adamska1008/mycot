# mycot

My implementation of some Chain-of-Thought solver. Currently implement zero-shot solver(baseline) and Plan-and-Solve solver(paper [here](https://arxiv.org/abs/2305.04091)).

## Usage

`main.py` is the script to run. Run `python main.py -h` to see available options:

```
usage: main.py [-h] --solver {zero_shot,plan_and_solve,give_a_list} [{zero_shot,plan_and_solve,give_a_list} ...] --dataset {AddSub,GSM8K,AQuA} [{AddSub,GSM8K,AQuA} ...] [--debug] [--range RANGE]

Use this script to quickly test the effect of a Solver solving a problem in the dataset

options:
  -h, --help            show this help message and exit
  --solver {zero_shot,plan_and_solve,give_a_list} [{zero_shot,plan_and_solve,give_a_list} ...]
                        Solver to be tested
  --dataset {AddSub,GSM8K,AQuA} [{AddSub,GSM8K,AQuA} ...]
                        Dataset to be tested
  --debug               set logger to debug level
  --range RANGE         Range of the problems to be tested
```

For example, run `python main.py --solver zero_shot --dataset AddSub` will execute the zero-shot solver(using prompt "let's think step by step") on the AddSub dataset. 

### Run Multi-Tests at the same time

The `solver` and `dataset` arguments accept multiple inputs. If you provide so, the script will create a cartesian product of the two lists and run all tests concurrently. The output will be stored to `./logs/${solver}_${dataset}.log`. For example, running

```sh
python main.py --solver zero_shot plan_and_solve --dataset AddSub AQuA
```

will initiate four threads, which are

```
(zero_shot, AddSub)
(plan_and_solve, AddSub)
(zero_shot, AQuA)
(plan_and_solve, AQuA)
```

### `--range`

The `--range` argument accepts a range argument in format of `start,end` (`end` not included) or a single `index`. For example, run

```sh
python main.py --solver zero_shot --dataset AddSub --range 2,3 # or --range 2
```

will test the problem with index `2` in AddSub dataset. 

### Extensibility

To add new datasets or CoTSolver, you will need to modify the code. It includes add new classes and change the codes related to arg-parse in `main.py`.

#### Adding a dataset

Currently the script supports loading data from JSON or JSONL files. Place your dataset file at `dataset` directory and create a new class in `loader.py`. It should implement `Problem` ABC if the answer is a number. It should implement `MultiChoiceProblem` if the answer is an option.

For example:

```python
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
    
    @classmethod
    def file_format(cls) -> str:
        return "json"
```

```python
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

    def options(self) -> dict[str, str]:
        res = {}
        for index, option in enumerate(self.raw_options):
            letter = chr(index + 65)
            while option.startswith(f"{letter}("):
                option = option[2:]
            res[letter] = option
        return res

    @classmethod
    def file_format(cls) -> str:
        return "jsonl"
```

#### Adding a Solver

Create a new class in `solver.py`. It should implement the `CoTSolver` class. For example:

```python
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
```

## Dependencies

### LLM

This repo depends on OpenAI API. Make sure you set your OpenAI API key and base url correctly.

For Ollama I use llama3:8b. Run after install ollama:

```sh
ollama pull llama3
```

### pip

```sh
pip install -r requirements.txt
```

## Conclusion

PS and PS+ seems to be useless. The behave basically the same as zero-shot(e.g. add "AI: Let's think step by step"). Wait, actually it's even worse when run on llama3:8b.

### gpt-4o-mini

|                | AddSub | GSM8K  | AQuA   |
| -------------- | ------ | ------ | ------ |
| Plan-and-Solve | 99.24% | 95.33% | 82.28% |
| zero-shot      | 99.24% | 95.67% | 80.31% |

### llama3:8b

|                | AddSub | GSM8K  | AQuA   |
| -------------- | ------ | ------ | ------ |
| Plan-and-Solve | 77.72% | 25.67% | 29.52% |
| zero-shot      | 87.34% | 68.33% | 50%    |

Man! What can I say.
