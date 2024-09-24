# mycot

My implementation of some Chain-of-Thought solver. Currently implement zero-shot solver(baseline) and Plan-and-Solve solver(paper [here](https://arxiv.org/abs/2305.04091)).

## Build

### LLM

This repo depends on OpenAI API. Make sure you set your OpenAI API key and base url correctly.

For Ollama I use llama3:8b. Run after install ollama:

```sh
ollama pull llama3
```

### Conda

```sh
conda env create -f environment.yml
conda activate mycot
```

## Run

Run `evaluate.py` will create a logfile and shows the comparison of two solver on *AddSub* Dataset.

## Conclusion

PS and PS+ seems to be useless. The behave basically the same as zero-shot(e.g. add "AI: Let's think step by step").

### gpt-4o-mini

|                | AddSub | GSM8K  | AQuA   |
| -------------- | ------ | ------ | ------ |
| Plan-and-Solve | 99.24% | 95.33% | 82.28% |
| zero-shot      | 99.24% | 95.67% | 80.31% |

### llama3:b

|                | AddSub | GSM8K | AQuA |
| -------------- | ------ | ----- | ---- |
| Plan-and-Solve | 77.72% |       |      |
| zero-shot      | 87.34% |       |      |
