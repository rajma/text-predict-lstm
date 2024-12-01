import torch
import sys

from rnn_scratch.data_prep import word_to_tensor, ingest_data

rnn = torch.load('char-rnn-classification.pt', weights_only=False)



# Just return an output given a word
def evaluate(word_tensor):
    hidden = rnn.init_hidden()

    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)

    return output


def predict(word:str, all_categories:list[str], n_predictions:int=3):
    output = evaluate(word_to_tensor(word))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions


if __name__ == '__main__':
    cat_data = ingest_data("/home/annaic/dev/rajma/text-predict-lstm/data/names/*.txt")
    print('>Dovesky')
    predict('Dovesky', cat_data.categories)
    print('>Jackson')
    predict('Jackson', cat_data.categories)
    print('>Satoshi')
    predict('Satoshi', cat_data.categories)
