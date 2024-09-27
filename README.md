# mycot

My implementation of some Chain-of-Thought solver. Currently implement zero-shot solver(baseline) and Plan-and-Solve solver(paper [here](https://arxiv.org/abs/2305.04091)).

## Usage

`main.py` is the script to run. Run `python main.py -h` and you will see:

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

For example, run `python main.py --solver zero_shot --dataset AddSub` will run the zero-shot solver(which use prompt "let's think step by step") on AddSub dataset. 

### Run Multi-Tests at the same time

The `solver` and `dataset` arguments accept multiple inputs. If you give so, the script will make a cartesian product on the two list and run all the tests with multi-thread. The output will be stored to `./logs/${solver}_${dataset}.log`. For example, run with

```sh
python main.py --solver zero_shot plan_and_solve --dataset AddSub AQuA
```

will run four threads, which are

```
(zero_shot, AddSub)
(plan_and_solve, AddSub)
(zero_shot, AQuA)
(plan_and_solve, AQuA)
```

### `--range`

The `--range` accepts a range argument in format of `start,end` (`end` not included) or `index`. For example, run

```sh
python main.py --solver zero_shot --dataset AddSub --range 2,3 # or --range 2
```

will test the problem with index `2` in AddSub dataset. 

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
