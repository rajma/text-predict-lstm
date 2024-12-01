import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rnn_scratch.data_prep import (
    ALL_ASCII_LETTERS,
    ingest_data,
    letter_to_tensor,
    word_to_tensor,
)


class RNN(nn.Module):
    """This RNN module implements a “vanilla RNN” an is just 3 linear layers which operate on an input and
    hidden state, with a LogSoftmax layer after the output."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # convert from num features in output to num features in hidden
        self.i2h = nn.Linear(input_size, hidden_size)
        # layer for forward pass
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # convert from num features in hidden to num features in output
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self) -> Tensor:
        # initialize the hidden state with zeroes
        # one dim vector of hidden_size variables
        return torch.zeros(1, self.hidden_size)

    def forward(self, input_x: Tensor, hidden: Tensor) -> (Tensor, Tensor):
        # new hidden is calculated with new input and last hidden
        hidden = F.tanh(self.i2h(input_x) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden


def category_from_output(output: Tensor, all_categories: list[str]) -> (str, int):
    top_values, top_indices = output.topk(1)
    category_index = top_indices[0].item()
    return all_categories[category_index], category_index


if __name__ == "__main__":
    n_letters = len(ALL_ASCII_LETTERS)
    cat_data = ingest_data("/home/annaic/dev/rajma/text-predict-lstm/data/names/*.txt")
    n_categories = cat_data.n_categories
    print(n_categories)
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    input_data = word_to_tensor("Albert")
    print(input_data[0])
    hidden = rnn.init_hidden()
    output, hidden = rnn(input_data[0], hidden)
    print(output)
    print(output.topk(1))
    print(cat_data.categories)
    print(category_from_output(output, cat_data.categories))
