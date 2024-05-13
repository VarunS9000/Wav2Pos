
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
import random
import pickle

with open('POS_data/train_data.pkl', 'rb') as f: # audio embeddings pulled out of fine tuned wav2vec2 for training set
    X_train = pickle.load(f)

with open('POS_data/test_data.pkl', 'rb') as f: # audio embeddings pulled out of fine tuned wav2vec2 for validation set
    X_test = pickle.load(f)

with open('span_normalized/train_segments.pkl', 'rb') as f:
      index_tuples_train = pickle.load(f)

with open('span_normalized/test_segments.pkl', 'rb') as f:
      index_tuples_test = pickle.load(f)

with open('span_normalized/train_gold.pkl', 'rb') as f:
    train_gold = pickle.load(f)

with open('span_normalized/test_gold.pkl', 'rb') as f:
    test_gold = pickle.load(f)

with open('span_normalized/train_predictions.pkl', 'rb') as f:
    train_predictions = pickle.load(f)

with open('span_normalized/test_predictions.pkl', 'rb') as f:
    test_predictions = pickle.load(f)

with open('span_normalized/predictions_train.pkl', 'rb') as f:
    predictions_train = pickle.load(f)

with open('span_normalized/predictions_test.pkl', 'rb') as f:
    predictions_test = pickle.load(f)

with open('span_normalized3/Y_train.pkl', 'rb') as f:
    Y_train_pre = pickle.load(f)

with open('span_normalized3/Y_test.pkl', 'rb') as f:
    Y_test_pre = pickle.load(f)

class ForcedAlignmentCTCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(ForcedAlignmentCTCLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 2, bidirectional = True, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = log_softmax(x,dim=-1)
        return x

with open('data/tag_to_ix0.pkl', 'rb') as f:  # dictionary of where key is a tag and value is number representing the tag
    tag_to_ix = pickle.load(f)

Y_train = []

for y in Y_train_pre:
  list_ = []
  for tup in y[0]:
    list_.append(tag_to_ix[tup[0]])
  Y_train.append(torch.tensor(list_))

Y_test = []

for y in Y_test_pre:
  list_ = []
  for tup in y[0]:
    list_.append(tag_to_ix[tup[0]])
  Y_test.append(torch.tensor(list_))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix.keys()),0)
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)
optimizer = torch.optim.Adam(lstm_pos.parameters(), lr=0.0001)
lstm_pos = lstm_pos.to(device)
ctc_loss = ctc_loss.to(device)
num_ = len(X_train)
num_test = len(X_test)

import pickle
with open('span_normalized/train_segments.pkl', 'rb') as f: # list of tuples for each training data point where each tuple corresponds to the first and last index for the corresponding word
    train_word_segments = pickle.load(f)

with open('span_normalized/test_segments.pkl', 'rb') as f: # list of tuples for each validation data point where each tuple corresponds to the first and last index for the corresponding word
    test_word_segments = pickle.load(f)

import torch
import torch.nn as nn

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Linear layer to map hidden state to output
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):

        # Encoder forward pass
        _, enc_hidden = self.encoder(x)
        # Initialize hidden states for decoder
        dec_hidden = enc_hidden
        # Decoder forward pass
        output, _ = self.decoder(x, dec_hidden)
        # Linear layer to map decoder output to final representation
        hidden_representation = self.linear(output)
        return hidden_representation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = LSTMEncoderDecoder(1024, 1024,2)
model1 = model1.to(device)

from tqdm import tqdm

def calculate_val_loss(model,data_test,y_test):
  count = 0
  total_loss = 0
  model.eval()
  model1.eval()
  with torch.no_grad():
    for b in tqdm(data_test):

      sentence = []
      for j in range(len(test_word_segments[count])):
        out, (h,c) = model1.encoder(b[test_word_segments[count][j][0]:test_word_segments[count][j][1]+1])
        sentence.append(h[0])

      word_embeddings = torch.stack(sentence)
      tags = Y_test[count]

      input_length = torch.tensor(len(word_embeddings))
      target_length = torch.tensor(len(tags))

      input_length = input_length.to(device)
      target_length = target_length.to(device)


      word_embeddings = word_embeddings.to(device)
      tags = tags.to(device)

      output = lstm_pos(word_embeddings)

      loss = ctc_loss(output, tags,input_length,target_length)

      total_loss+= loss.item()

      count +=1

  return total_loss

def generate_lcs1(list1, list2):
    m = len(list1)
    n = len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    prev = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1][0] == list2[j - 1][0]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                prev[i][j] = 'diagonal'
            else:
                if dp[i - 1][j] >= dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j]
                    prev[i][j] = 'up'
                else:
                    dp[i][j] = dp[i][j - 1]
                    prev[i][j] = 'left'

    lcs_words = []
    i = m
    j = n
    while i > 0 and j > 0:
        if prev[i][j] == 'diagonal':
            lcs_words.append(list1[i - 1][0])
            i -= 1
            j -= 1
        elif prev[i][j] == 'up':
            i -= 1
        else:
            j -= 1

    lcs_words.reverse()

    return lcs_words

def generate_lcs2(list1, list2):
    m = len(list1)
    n = len(list2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    prev = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1][0] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                prev[i][j] = 'diagonal'
            else:
                if dp[i - 1][j] >= dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j]
                    prev[i][j] = 'up'
                else:
                    dp[i][j] = dp[i][j - 1]
                    prev[i][j] = 'left'

    lcs_sequence = []
    i = m
    j = n
    while i > 0 and j > 0:
        if prev[i][j] == 'diagonal':
            lcs_sequence.append(list1[i - 1])
            i -= 1
            j -= 1
        elif prev[i][j] == 'up':
            i -= 1
        else:
            j -= 1

    lcs_sequence.reverse()

    return lcs_sequence


with open('span_normalized/train_gold.pkl', 'rb') as f: # Gold transcriptions of the training data
    train_gold = pickle.load(f)

with open('span_normalized/test_gold.pkl', 'rb') as f: # Gold transcriptions of the validation data
    test_gold = pickle.load(f)

actuals_train = []
actuals_test = []

for i in range(len(train_gold)):
  temp = []
  x = train_gold[i].split()
  for j in range(len(x)):

    temp.append((x[j],Y_train[i][j]))

  actuals_train.append(temp)

for i in range(len(test_gold)):
  temp = []
  x = test_gold[i].split()

  for j in range(len(x)):
    temp.append((x[j],Y_test[i][j]))

  actuals_test.append(temp)

with open('span_normalized/predictions_train.pkl', 'rb') as f:  # predictions of the Wav2Vec2ASR on the train data
      train_predictions = pickle.load(f)

with open('span_normalized/predictions_test.pkl', 'rb') as f:  # predictions of the Wav2Vec2ASR on the validation data
      test_predictions = pickle.load(f)

def get_accuracy(model, train_sent_predictions, test_sent_predictions, X_train, X_test):
  

  with torch.no_grad():
    predictions_train = []
    predictions_test = []

    count = 0
    for b in tqdm(X_train):
      
      sentence = []
      for j in range(len(train_word_segments[count])):
        out, (h,c) = model1.encoder(b[train_word_segments[count][j][0]:train_word_segments[count][j][1]+1])
        sentence.append(h[0])

      word_embeddings = torch.stack(sentence)
      tags = Y_train[count]

      input_length = torch.tensor(len(word_embeddings))
      target_length = torch.tensor(len(tags))

      input_length = input_length.to(device)
      target_length = target_length.to(device)


      word_embeddings = word_embeddings.to(device)
      tags = tags.to(device)

      output = lstm_pos(word_embeddings)
      _, indices = torch.max(output,1)
      ret = []
      for i in range(len(indices)):
          for key, value in tag_to_ix.items():
              if indices[i] == value:
                  ret.append(value)
      temp = []
      x = train_sent_predictions[count].split()
      for i in range(len(b)):
        temp.append((x[i],ret[i]))

      predictions_train.append(temp)

      count+=1

    count = 0
    for b in tqdm(X_test):
      
      sentence = []
      for j in range(len(train_word_segments[count])):
        out, (h,c) = model1.encoder(b[train_word_segments[count][j][0]:train_word_segments[count][j][1]+1])
        sentence.append(h[0])

      word_embeddings = torch.stack(sentence)
      tags = Y_test[count]

      input_length = torch.tensor(len(word_embeddings))
      target_length = torch.tensor(len(tags))

      input_length = input_length.to(device)
      target_length = target_length.to(device)


      word_embeddings = word_embeddings.to(device)
      tags = tags.to(device)

      output = lstm_pos(word_embeddings)
      _, indices = torch.max(output,1)
      ret = []
      for i in range(len(indices)):
          for key, value in tag_to_ix.items():
              if indices[i] == value:
                  ret.append(value)
      temp = []
      x = test_sent_predictions[count].split()
      for i in range(len(b)):
        temp.append((x[i],ret[i]))

      predictions_test.append(temp)

      count+=1




    couples_train = []
    for i in range(len(actuals_train)):

      gen1 = generate_lcs1(actuals_train[i],predictions_train[i])
      gen20 = generate_lcs2(predictions_train[i],gen1)
      gen21 = generate_lcs2(actuals_train[i],gen1)
      couples_train.append((gen20,gen21))

    couples_test = []
    for i in range(len(actuals_test)):

      gen1 = generate_lcs1(actuals_test[i],predictions_test[i])
      gen20 = generate_lcs2(predictions_test[i],gen1)
      gen21 = generate_lcs2(actuals_test[i],gen1)
      couples_test.append((gen20,gen21))

    total_train = 0
    total_test = 0

    for c in couples_train:
      total_train+=len(c[0])

    for c in couples_test:
      total_test+=len(c[0])

    correct_train = 0
    for tup in couples_train:
      l1 = tup[0]
      l2 = tup[1]

      for i in range(len(l1)):
        if l1[i][1] == l2[i][1]:
          correct_train += 1

    train_accuracy = correct_train/total_train

    correct_test = 0

    for tup in couples_test:
      l1 = tup[0]
      l2 = tup[1]

      for i in range(len(l1)):
        if l1[i][1] == l2[i][1]:
          correct_test += 1

    test_accuracy = correct_test/total_test


  return train_accuracy, test_accuracy


def train2(lstm_pos, dropout):
  train_losses = []
  test_losses = []
  train_accuracies = []
  test_accuracies = []
  for epoch in range(10):
      total_loss = 0
      count = 0
      lstm_pos.train()
      model1.train()
      for b in tqdm(X_train):

        sentence = []
        for j in range(len(train_word_segments[count])):
          out, (h,c) = model1.encoder(b[train_word_segments[count][j][0]:train_word_segments[count][j][1]+1])
          sentence.append(h[0])

        word_embeddings = torch.stack(sentence)
        tags = Y_train[count]

        input_length = torch.tensor(len(word_embeddings))
        target_length = torch.tensor(len(tags))

        input_length = input_length.to(device)
        target_length = target_length.to(device)


        word_embeddings = word_embeddings.to(device)
        tags = tags.to(device)

        output = lstm_pos(word_embeddings)

        loss = ctc_loss(output, tags,input_length,target_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()

        count+=1

      total_loss_test = calculate_val_loss(lstm_pos,X_test,Y_test)
      print('Epoch',epoch)
      print('Train Loss ',total_loss/num_)
      print('Test Loss ',total_loss_test/num_test)


      train_losses.append(total_loss)
      test_losses.append(total_loss_test)

      torch.save({
              'model_state_dict': lstm_pos.state_dict(),
              'encoder_state_dict': model1.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss_list_train': train_losses,
              'loss_list_test': test_losses,
              'train_accuracies': train_accuracies,
              'test_accuracies': test_accuracies,
              'epoch': epoch,
              }, f"Part1/models/lstm_pos_"+str(epoch)+".pt")

dropouts = [3]

# 3: 7

for d in dropouts:
  drop = d/10
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix.keys()),drop)
  model1 = LSTMEncoderDecoder(1024, 1024,2)
  ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)
  optimizer = torch.optim.Adam(lstm_pos.parameters(), lr=0.0001)
  lstm_pos = lstm_pos.to(device)
  ctc_loss = ctc_loss.to(device)
  lstm_pos.train()
  model1.train()
  num_ = len(X_train)
  num_test = len(X_test)
  print(f"Dropout {drop} begins")
  train2(lstm_pos,d)
  print(f"Dropout {drop} ends")
  print()

# 8, 65.7, 61.7
dropout_list = [3]
for t in dropout_list:
  print(f'Dropout {t}')
  for e in range(10):
    checkpoint = torch.load(f"Part1/models/lstm_pos_{e}.pt")
    lstm_pos.load_state_dict(checkpoint['model_state_dict'])
    lstm_pos.eval()
    print(f"Epoch {e}")
    t1, t2 = get_accuracy(lstm_pos, train_predictions, test_predictions, X_train, X_test)
    print(f"Train Accuracy {t1}")
    print(f"Test Accuracy {t2}")
    print()

  print(f"Dropout {t} complete")

