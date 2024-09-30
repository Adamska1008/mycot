import json
from random import randint, random

PROBLEM_NUMBER = 200
DESCRIPTION_TEMPLATE = "How many {} are there in the word {}?"
WORD_LEN_UPPER_LIMIT = 40
WORD_LEN_LOWER_LIMIT = 20

def random_word(length):
    return [chr(randint(ord('a'), ord('z'))) for _ in range(length)]

problems = []

for i in range(PROBLEM_NUMBER):
    index = i
    word_len = randint(WORD_LEN_LOWER_LIMIT, WORD_LEN_UPPER_LIMIT + 1)
    word = random_word(word_len)
    target_letter = chr(randint(ord('a'), ord('z')))
    turn_rate = random() / 2
    for i, _ in enumerate(word):
        dice = random()
        if dice <= turn_rate:
            word[i] = target_letter
    word = ''.join(word)
    print(word)
    letter_cnt = word.count(target_letter)
    problem = {
        "index": index, 
        "problem": DESCRIPTION_TEMPLATE.format(target_letter, word),
        "answer": letter_cnt,
    }
    problems.append(problem)

with open("howmany.json", "w", encoding="utf-8") as json_file:
    json.dump(problems, json_file, indent=4) 
