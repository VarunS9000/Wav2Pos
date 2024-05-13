import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
import pickle

with open('POS_data/train_data.pkl', 'rb') as f:  # audio embeddings pulled out of fine tuned wav2vec2 for training set
    X_train = pickle.load(f)
    
with open('POS_data/test_data.pkl', 'rb') as f: # audio embeddings pulled out of fine tuned wav2vec2 for validation set
    X_test = pickle.load(f)

with open('span_normalized3/Y_train.pkl', 'rb') as f:
    Y_train_pre = pickle.load(f)

with open('span_normalized3/Y_test.pkl', 'rb') as f:
    Y_test_pre = pickle.load(f)


class ForcedAlignmentCTCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout):
        super(ForcedAlignmentCTCLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 2, bidirectional = True, batch_first=True, dropout = dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = log_softmax(x,dim=-1)
        return x

def prepare_seq(tags,i2x):
  seq = []
  for t in tags:
    for _ in range(t[1]):
      seq.append(i2x[t[0]])
    seq.append(i2x[' '])

  seq.pop()

  return torch.tensor(seq, dtype=torch.long)

def create_tuples(data_list):
  data_list = data_list[0]
  tups = []
  start = 0
  for d in data_list:
    tups.append((start,d['end_offset']))
    start = d['end_offset'] + 1

  return tups

with open('drive/MyDrive/span_normalized3/tag_to_ix.pkl', 'rb') as f: # vocabulary dictionary for POS tagger
    tag_to_ix = pickle.load(f)


Y_train = [prepare_seq(y[0],tag_to_ix) for y in Y_train_pre]
Y_test = [prepare_seq(y[0],tag_to_ix) for y in Y_test_pre]

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
  
"""
with open('drive/MyDrive/span_normalized/train_segments.pkl', 'rb') as f: # List of tuples for each of the train data, where the tuple contains the first and last index of the audio embedding that corresponds to a word
      index_tuples_train = pickle.load(f)

with open('drive/MyDrive/span_normalized/test_segments.pkl', 'rb') as f: # List of tuples for each of the validation data, where the tuple contains the first and last index of the audio embedding that corresponds to a word
      index_tuples_test = pickle.load(f)
      

"""

index_tuples_train = [create_tuples(t[-1]) for t in X_train] # List of tuples for each of the train data, where the tuple contains the first and last index of the audio embedding that corresponds to a word
index_tuples_test = [create_tuples(t[-1]) for t in X_test] # List of tuples for each of the validation data, where the tuple contains the first and last index of the audio embedding that corresponds to a word

with open('drive/MyDrive/span_normalized/train_gold.pkl', 'rb') as f: # Gold transcription for training data
      train_gold = pickle.load(f)

with open('drive/MyDrive/span_normalized/test_gold.pkl', 'rb') as f: # Gold transcription for validation data
      test_gold = pickle.load(f)

actuals_train = []
actuals_test = []

for i in range(len(Y_train_pre)):
  t0 = Y_train_pre[i][1].split()
  t1 = Y_train_pre[i][0]
  t2 = train_gold[i].split()
  temp = []

  for j in range(len(t1)):
    temp.append((t2[j],t1[j][0]))
  actuals_train.append(temp)

for i in range(len(Y_test_pre)):
  t1 = Y_test_pre[i][0]
  t2 = test_gold[i].split()
  temp = []
  for j in range(len(t1)):
    temp.append((t2[j],t1[j][0]))
  actuals_test.append(temp)

with open('drive/MyDrive/span_normalized/predictions_train.pkl', 'rb') as f: # predictions of the Wav2Vec2ASR on the train data
      train_predictions = pickle.load(f)

with open('drive/MyDrive/span_normalized/predictions_test.pkl', 'rb') as f: # predictions of the Wav2Vec2ASR on the validation data
      test_predictions = pickle.load(f)

from collections import Counter

def top_occurrences(input_list):
    # Count occurrences of each element in the list
    element_counts = Counter(input_list)

    # Get the top 3 occurring elements and their counts
    top_elements = element_counts.most_common(3)

    return top_elements
def get_accuracy(model, train_sent_predictions, test_sent_predictions, X_train, X_test):

  predictions_train = []
  predictions_test = []


  c = 0
  print('Fetching train results')
  with torch.no_grad():
    for b in tqdm(X_train):

      embeddings = b[2][0]
      tags = Y_train[c]

      input_length = torch.tensor(len(embeddings))
      target_length = torch.tensor(len(tags))

      input_length = input_length.to(device)
      target_length = target_length.to(device)


      embeddings = embeddings.to(device)
      tags = tags.to(device)

      output = lstm_pos(embeddings)
      _, indices = torch.max(output,1)
      ret = []
      for i in range(len(indices)):
          for key, value in tag_to_ix.items():
              if indices[i] == value:
                  ret.append(key)

      new_ret_train = []

      for tup in index_tuples_train[c]:
        a = ret[tup[0]:tup[1]+1]
        top_occ = top_occurrences(a)
        z = '-'
        for t in top_occ:
          if t[0] not in ['-',' ']:
             z = t[0]
             break
        new_ret_train.append(z)
        z = '-'

      temp = []
      x = train_sent_predictions[c].split()
      for i in range(len(new_ret_train)):
        temp.append((x[i],new_ret_train[i]))

      predictions_train.append(temp)



      c+=1

  c = 0
  print()
  print('Fetching Test Results')
  with torch.no_grad():
    for b in tqdm(X_test):

      embeddings = b[2][0]
      tags = Y_test[c]

      input_length = torch.tensor(len(embeddings))
      target_length = torch.tensor(len(tags))

      input_length = input_length.to(device)
      target_length = target_length.to(device)


      embeddings = embeddings.to(device)
      tags = tags.to(device)

      output = lstm_pos(embeddings)
      _, indices = torch.max(output,1)
      ret = []
      for i in range(len(indices)):
          for key, value in tag_to_ix.items():
              if indices[i] == value:
                  ret.append(key)

      new_ret_test = []

      for tup in index_tuples_test[c]:
        a = ret[tup[0]:tup[1]+1]
        top_occ = top_occurrences(a)
        z = '-'
        for t in top_occ:
          if t[0] not in ['-',' ']:
             z = t[0]
             break
        new_ret_test.append(z)


      temp = []
      x = test_sent_predictions[c].split()
      for i in range(len(new_ret_test)):
        temp.append((x[i],new_ret_test[i]))

      predictions_test.append(temp)

      c+=1

  print('Generating train subsequences')
  couples_train = []
  for i in tqdm(range(len(actuals_train))):

    gen1 = generate_lcs1(actuals_train[i],predictions_train[i])
    gen20 = generate_lcs2(predictions_train[i],gen1)
    gen21 = generate_lcs2(actuals_train[i],gen1)
    couples_train.append((gen20,gen21))

  print()
  print('Generating test subsequences')
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
  total_test = 0
  for c in couples_test:
    total_test+=len(c[0])
  test_accuracy = correct_test/total_test

  print("Train Accuracy ", train_accuracy)
  print("Test Accuracy ", test_accuracy)


  return predictions_train, actuals_train, predictions_test, actuals_test


from tqdm import tqdm
def calculate_val_loss(model,data_test,y_test):
  count = 0
  total_loss = 0
  model.eval()
  with torch.no_grad():
    for b in tqdm(data_test):
      embeddings = b[2][0]
      tags = Y_test[count]

      input_length = torch.tensor(len(embeddings))
      target_length = torch.tensor(len(tags))

      input_length = input_length.to(device)
      target_length = target_length.to(device)


      embeddings = embeddings.to(device)
      tags = tags.to(device)

      output = lstm_pos(embeddings)

      loss = ctc_loss(output, tags,input_length,target_length)

      total_loss+= loss.item()

      count +=1

  return total_loss


test_map = {1:[], 2:[], 3:[], 4:[], 5:[]}

def train(lstm_pos, dropout):
  train_losses = []
  test_losses = []
  train_accuracies = []
  test_accuracies = []
  for epoch in range(31):
      total_loss = 0
      count = 0
      lstm_pos.train()
      for b in tqdm(X_train):
        embeddings = b[2][0]
        tags = Y_train[count]

        input_length = torch.tensor(len(embeddings))
        target_length = torch.tensor(len(tags))

        input_length = input_length.to(device)
        target_length = target_length.to(device)


        embeddings = embeddings.to(device)
        tags = tags.to(device)

        output = lstm_pos(embeddings)

        loss = ctc_loss(output, tags,input_length,target_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()

        count +=1

      total_loss_test = calculate_val_loss(lstm_pos,X_test,Y_test)
      print('Epoch',epoch)
      print('Train Loss ',total_loss/num_)
      print('Test Loss ',total_loss_test/num_test)
      
      if epoch> 0 and epoch%3==0:
        torch.save({
                'model_state_dict': lstm_pos.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_list_train': train_losses,
                'loss_list_test': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
                'epoch': epoch,
                }, f"drive/MyDrive/span_normalized3/Part2/lstm_pos_"+str(epoch)+".pt")


dropouts = [2] # 0.2 dropout geve the best results

for d in dropouts:
  drop = d/10
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix.keys()),0.2)
  ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)
  optimizer = torch.optim.Adam(lstm_pos.parameters(), lr=0.0001)
  lstm_pos = lstm_pos.to(device)
  ctc_loss = ctc_loss.to(device)
  lstm_pos.train()
  num_ = len(X_train)
  num_test = len(X_test)
  print(f"Dropout {drop} begins")
  train(lstm_pos,d)
  print(f"Dropout {drop} ends")
  print()

for i in range(8,10):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(f"drive/MyDrive/span_normalized2/Part2/lstm_pos_{3*i}.pt")
  lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix.keys()),0.2)
  lstm_pos.load_state_dict(checkpoint['model_state_dict'])
  lstm_pos = lstm_pos.to(device)
  lstm_pos.eval()
  print(f'Fetching accuracies for epoch {3*i}')
  a,b,c,d = get_accuracy(lstm_pos, train_predictions, test_predictions, X_train, X_test)


from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(f"drive/MyDrive/Part2/lstm_pos_28.pt")
lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix.keys()),0.2)
lstm_pos.load_state_dict(checkpoint['model_state_dict'])
lstm_pos = lstm_pos.to(device)
lstm_pos.eval()

print(f'Fetching accuracies for epoch 28')
p1,a1,p2,a2 = get_accuracy(lstm_pos, train_predictions, test_predictions, X_train, X_test)

