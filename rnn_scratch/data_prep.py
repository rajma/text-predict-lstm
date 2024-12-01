import string
from io import open
import glob
import os
import unicodedata
import random
from typing import Any

import torch
from torch import Tensor

from rnn_scratch.schema import CategoryData


def list_data_files(pathname: str):
    return glob.glob(pathname)


ALL_ASCII_LETTERS = string.ascii_letters + " .,;'"


def unicode_to_ascii(s: str):
    """Turn a Unicode string to plain ASCII from stackoverflow"""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in ALL_ASCII_LETTERS
    )


def read_lines(filename: str):
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


def ingest_data(path_pattern: str) -> CategoryData:
    categories = []
    category_lines: dict[str, list[str]] = {}
    for file_name in list_data_files(path_pattern):
        # print(file_name)
        # print(os.path.splitext(os.path.basename(file_name)))
        category = os.path.splitext(os.path.basename(file_name))[0]
        categories.append(category)
        category_lines[category] = read_lines(file_name)

    return CategoryData(categories=categories, category_lines=category_lines)


def letter_to_index(letter: str) -> int:
    return ALL_ASCII_LETTERS.find(letter)


def letter_to_tensor(letter: str) -> Tensor:
    # 1 row, len of vocab columns
    tensor = torch.zeros(1, len(ALL_ASCII_LETTERS))
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def word_to_tensor(word: str) -> Tensor:
    tensor = torch.zeros(
        len(word), 1, len(ALL_ASCII_LETTERS)
    )  # 1 represents batch size
    for i, letter in enumerate(word):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_choice(choices: list[Any]) -> Any:
    return random.choice(choices)


def random_training_pair(category_data: CategoryData) -> (str, str, Tensor, Tensor):
    category = random_choice(category_data.categories)
    word_choice = random_choice(category_data.category_lines[category])
    category_tensor = torch.tensor(
        [category_data.categories.index(category)], dtype=torch.long
    )
    word_tensor = word_to_tensor(word_choice)
    return category, word_choice, category_tensor, word_tensor


if __name__ == "__main__":
    # print(unicode_to_ascii('Ślusàrski'))
    cat_data = ingest_data("/home/annaic/dev/rajma/text-predict-lstm/data/names/*.txt")
    print(cat_data.category_lines["French"][:5])
    print(len(ALL_ASCII_LETTERS))
    abc_tensor = word_to_tensor("abc")
    print(abc_tensor.size())
    print(abc_tensor[1])
    print(f"Number of characters in word: {abc_tensor.size()[0]}")
    for i in range(abc_tensor.size()[0]):
        print(abc_tensor[i].shape)  # tensor for each character

    # c,w,c_t,w_t = random_training_pair(cat_data)
    # print('category =', c, '/ word =', w, 'c_t=', c_t, 'w_t=', w_t)
