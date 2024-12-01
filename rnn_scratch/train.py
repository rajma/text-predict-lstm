import time
import math

import torch
import torch.nn as nn
from torch import Tensor

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from rnn_scratch.data_prep import ALL_ASCII_LETTERS, ingest_data, random_training_pair
from rnn_scratch.rnn_model import RNN, category_from_output

n_letters = len(ALL_ASCII_LETTERS)
n_categories = 18
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# Negative log likelihood loss
criterion = nn.NLLLoss()
learning_rate = (
    0.005  # If you set this too high, it might explode. If too low, it might not learn
)

"""
- Create input and target tensors
- Create a zeroed initial hidden state
- Read each letter in and
- Keep hidden state for next letter
- Compare final output to target
- Back-propagate
- Return the output and loss
"""


def train(category_tensor: Tensor, word_tensor: Tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()  # reset gradients of all parameters to 0

    # initialize to prevent warnings
    output = torch.zeros(1, n_letters)

    # encoding tensor for each letter in word
    for i in range(word_tensor.shape[0]):
        # forward with last letter hidden and current letter encoding tensor
        output, hidden = rnn(word_tensor[i], hidden)

    loss = criterion(output, category_tensor)  # calculate the original loss
    loss.backward()  # recompute the loss from back-propagation

    # add parameter's gradients to their values multiplied by learning rate
    for p in rnn.parameters():
        # add_ is the inplace version of add
        # add gradient to the value
        p.data.add_(p.grad.data, alpha=-learning_rate)
        # we update the rnn with latest values for the next use

    return output, loss.item()  # loss item is a scalar


def time_since(since: float):
    now = time.time()
    s = now - since
    m = math.floor(s / 160)
    s -= m * 60
    return "%dm %ds" % (m, s)


n_epochs = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

if __name__ == "__main__":
    cat_data = ingest_data("/home/annaic/dev/rajma/text-predict-lstm/data/names/*.txt")

    start = time.time()
    for epoch in range(1, n_epochs + 1):
        category, word, cat_tensor, word_tensor = random_training_pair(cat_data)
        output, loss = train(cat_tensor, word_tensor)
        current_loss += loss

        if epoch % print_every == 0:
            guess, guess_idx = category_from_output(output, cat_data.categories)
            correct = "✓" if guess == category else "✗ (%s)" % category
            print(
                "%d %d%% (%s) %.4f %s / %s %s"
                % (
                    epoch,
                    epoch / n_epochs * 100,
                    time_since(start),
                    loss,
                    word,
                    guess,
                    correct,
                )
            )

            # Add the current loss avg to the list of losses
            if epoch % plot_every == 0:
                all_losses.append(current_loss/plot_every)
                current_loss = 0
    torch.save(rnn, 'char-rnn-classification.pt')
    plt.figure()
    plt.plot(all_losses)
