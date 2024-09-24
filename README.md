# mycot

My implementation of some Chain-of-Thought solver. Currently implement zero-shot solver(baseline) and Plan-and-Solve solver(paper [here](https://arxiv.org/abs/2305.04091)).

## Build

This repo depends on OpenAI API. Make sure you set your OpenAI API key and base url correctly.

### Conda

```sh
conda env create -f environment.yml
conda activate mycot
```

## Run

Run `evaluate.py` will create a logfile and shows the comparison of two solver on *AddSub* Dataset.

## Conclusion

PS and PS+ seems to be useless.

## TODOS

The accuracy seems all too high with OpenAI API. Try use ollama instead.
