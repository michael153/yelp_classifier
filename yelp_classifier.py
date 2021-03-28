"""Defines the methods that generate batches, BERT embeddings, and trains the
downstream Yelp model."""

import argparse
import os
import sys
import json
import pickle
import nltk
import torch
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from nltk import tokenize
from splitstream import splitfile
from transformers import AutoTokenizer, BertModel, BertConfig

from yelp_model import YelpModel

DATASET_PATH = os.path.dirname(os.path.abspath(__file__))

def setup_superbatches():
    """
    Setup superbatch directory to allow for fine-grained dataset loading. Assumes
    yelp_academic_dataset_review.json exists in root directory. This method
    is run if the "--setup" flag is passed.
    """
    superbatch = []
    superbatch_id = 0
    superbatch_size = 500
    max_batches = 500

    reviews_path = open(
        os.path.join(DATASET_PATH, "yelp_academic_dataset_review.json"))
    reviews_loader = tqdm(splitfile(reviews_path, format="json"),
                          position=0,
                          leave=True)
    reviews_loader.set_description("Loading superbatch %d..." % superbatch_id)

    for jsonstr in reviews_loader:
        superbatch.append(json.loads(jsonstr))
        if len(superbatch) >= superbatch_size:
            with open(
                    os.path.join(
                        DATASET_PATH,
                        "superbatches/yelp_superbatch_%d.pkl" % superbatch_id),
                    'wb') as fout:
                pickle.dump(superbatch, fout)
            superbatch = []
            superbatch_id += 1
            reviews_loader.set_description("Loading superbatch %d..." %
                                           superbatch_id)
            if superbatch_id > max_batches:
                break


def tokenize_review(review_str, bert_tokenizer, max_sentence_length=75):
    """
    Splits string into sentences, and then returns the tokenization of each
    consecutive pair of sentences

    Arguments:
        review_str (str): the review to be tokenized
        bert_tokenizer (transformers.AutoTokenizer): tokenizer object that
            generates BERT tokens
        max_sentence_length (int): max number of words per sentence
    """
    sentences = tokenize.sent_tokenize(review_str)
    tokenizer_params = sentences
    if len(sentences) > 1:
        tokenizer_params = list(zip(sentences, sentences[1:]))
    review_tokens = bert_tokenizer(tokenizer_params,
                                   padding='max_length',
                                   max_length=max_sentence_length,
                                   truncation=True,
                                   is_split_into_words=True,
                                   return_tensors="np")
    review_tokens['input_ids'] = torch.LongTensor(review_tokens['input_ids'])
    review_tokens['attention_mask'] = torch.FloatTensor(
        review_tokens['attention_mask'])
    review_tokens['token_type_ids'] = torch.LongTensor(
        review_tokens['token_type_ids'])
    return review_tokens


def generate_batch(superbatch_ids, bert_tokenizer):
    """
    Generates a batch of tokens and the respective relevant information
    for BERT to generate embeddings with. Because a single review may
    generate multiple tokens, the tokens_per_review list is needed

    Arguments:
        superbatch_ids (list): list of superbatch ids to load from
        bert_tokenizer (transformers.AutoTokenizer): tokenizer object that
            generates BERT tokens
    """
    stars = []  # y
    tokens_per_review = []
    batched_tokens = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': []
    }

    for superbatch_id in superbatch_ids:
        with open(
                os.path.join(DATASET_PATH,
                             "superbatches/yelp_superbatch_%d.pkl" % superbatch_id), 'rb') as f:
            batched_data = pickle.load(f)

        for review_entry in batched_data:
            review_tokens = tokenize_review(review_entry['text'], bert_tokenizer)
            tokens_per_review.append(review_tokens['input_ids'].shape[0])
            stars.append(review_entry['stars'])
            batched_tokens['input_ids'].append(review_tokens['input_ids'])
            batched_tokens['attention_mask'].append(
                review_tokens['attention_mask'])
            batched_tokens['token_type_ids'].append(
                review_tokens['token_type_ids'])

    batched_tokens['input_ids'] = torch.vstack(batched_tokens['input_ids'])
    batched_tokens['attention_mask'] = torch.vstack(
        batched_tokens['attention_mask'])
    batched_tokens['token_type_ids'] = torch.vstack(
        batched_tokens['token_type_ids'])
    return batched_tokens, stars, tokens_per_review


def generate_weighting_matrix(tokens_per_review):
    """
    Because a single review may generate multiple tokens / embeddings, our
    downstream model will need to calculate a weighted average of the
    activations, and will thus use this generated weight matrix.

    Arguments:
        tokens_per_review (list): A list of how many tokens each review uses
    """
    W = torch.zeros((len(tokens_per_review), sum(tokens_per_review)))
    running_sum = 0
    for row in range(len(tokens_per_review)):
        W[row, running_sum:running_sum +
          tokens_per_review[row]] = 1 / tokens_per_review[row]
        running_sum += tokens_per_review[row]
    return W


def generate_embeddings(bert_model, batch):
    """
    Generates the embeddings for a given batch

    Arguments:
        batch (dict): The generated batch dictionary that includes
            the keys input_ids, attention_mask, and token_type_ids
    """
    return bert_model.forward(**batch)


def get_current_minibatch_data(tokens_per_review, stars, bounds):
    """
    Calculates the weighted matrix and relevant target values
    for a minibatch defined by the starting and ending index (bounds)
    w.r.t the entire batch.

    Arguments:
        tokens_per_review (list): A list of how many tokens each review
            (in the batch) uses
        stars (list): The star ratings that correspond to the reviews
            in the batch
        bounds (tuple): The starting and ending indices that define the
            minibatch w.r.t to the batch
    """
    minibatch_size = sum(tokens_per_review[bounds[0]:bounds[1]])
    weighted_avg_matrix = generate_weighting_matrix(
        tokens_per_review[bounds[0]:bounds[1]])
    target = torch.FloatTensor(stars[bounds[0]:bounds[1]]) / 5
    return minibatch_size, weighted_avg_matrix, target


def train(yelp_model,
          optimizer,
          loss_fn,
          embeddings,
          stars,
          tokens_per_review,
          epochs=10):
    """
    Trains the downstream FC Yelp model.

    Arguments:
        yelp_model (torch.nn.Model): The FC yelp model
        optimizer (torch.optim.Optimizer): The model's optimizer
        loss_fn (torch.nn.Loss): The loss function for the model
        embeddings (torch.FloatTensor): The last hidden state embeddings
            generated by the upstream BERT model
        stars (list): The star ratings that correspond to the reviews in
            the batch
        tokens_per_review (list): A list of how many tokens each review
            (in the batch) uses
        epochs (int): Number of epochs to train for
    """
    for e in range(epochs):
        print("Epoch %d..." % e)
        iters = int(np.ceil(embeddings.shape[0] / 20))
        pos, embeddings_pos = 0, 0
        for i in range(iters):
            review_indices = (pos, min(pos + 20, len(tokens_per_review)))
            minibatch_size, weighted_avg_matrix, target = get_current_minibatch_data(
                tokens_per_review, stars, review_indices)
            pos += 20
            pred = yelp_model.forward(
                embeddings[embeddings_pos:embeddings_pos +
                           minibatch_size, :, :].detach().view(minibatch_size, -1))
            embeddings_pos += minibatch_size
            loss = loss_fn(
                torch.mm(weighted_avg_matrix, pred).flatten(), target)
            if i % 10 == 0:
                print("Epoch %d, Loss: %.3f" % (i, loss))
            loss.backward()
            optimizer.step()


def main():
    """Main method"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",
                        help="Download NLTK punkt corpus",
                        action="store_true")
    parser.add_argument("--setup",
                        help="Setup superbatch directory",
                        action="store_true")
    args = parser.parse_args()

    print("Path: %s\n" % DATASET_PATH)

    if args.corpus:
        print("Installing NLTK punkt...")
        nltk.download('punkt')

    if args.setup:
        setup_superbatches()

    print("Loading AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    print("Configuring and loading BERT model...")
    configuration = BertConfig()
    bert_model = BertModel(configuration)
    yelp_model = YelpModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(yelp_model.parameters(), lr=1e-2)

    print("Generating batch(es)...")
    batched_tokens, stars, tokens_per_review = generate_batch([0], tokenizer)

    print("Generating BERT embeddings...")
    embeddings = generate_embeddings(bert_model, batched_tokens)

    print("Training YELP model...\n")
    train(yelp_model, optimizer, loss_fn, embeddings[0], stars,
          tokens_per_review)
    torch.save(yelp_model.state_dict(),
               os.path.join(DATASET_PATH, 'yelp_model.pt'))


if __name__ == "__main__":
    main()
