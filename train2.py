

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(torch.__version__)

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 42 is blank space
def create_tuples(ids_tensor):
  start = 0
  end = 0
  ids = ids_tensor[0]
  n = len(ids)
  tups = []
  something_except_unk = False
  for i in range(n):
    if end == n-1:
      if something_except_unk:
        tups.append((start,end))
    elif ids[i]!=42:
      if ids[i]!=60:
        something_except_unk = True
      end+=1
    elif ids[i]==42:
      if i==n-1:
        if start!=end:
          tups.append((start,end))
      elif ids[i+1]!=42:
        if something_except_unk:
          tups.append((start,end))
          start = end + 1
          end = start
          something_except_unk = False
      else:
        end+=1

  return tups

import torchaudio
import torch
import librosa
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def speech_file_to_array_fn(audio):
    resampler = torchaudio.transforms.Resample(48_000, 16_000)
    return resampler(audio).squeeze().numpy()
def generate_speech_array(file_path):
  speech_array, sampling_rate = torchaudio.load(file_path)
  return speech_file_to_array_fn(speech_array)

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):

        _, enc_hidden = self.encoder(x)
        dec_hidden = enc_hidden
        output, _ = self.decoder(x, dec_hidden)
        hidden_representation = self.linear(output)
        return hidden_representation

model = LSTMEncoderDecoder(1024, 1024,2)
model = model.to(device)

from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ForcedAlignmentCTCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForcedAlignmentCTCLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 3, bidirectional = True, batch_first=True, dropout = 0.1)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = log_softmax(x,dim=-1)
        return x

tag_to_ix = {'-': 0, 'ADV': 1, 'NOUN': 2, 'AUX': 3, 'ADP': 4, '_': 5, 'SCONJ': 6, 'VERB': 7, 'PROPN': 8, 'PUNCT': 9, 'PRON': 10, 'X': 11, 'ADJ': 12, 'NUM': 13, 'INTJ': 14, 'DET': 15, 'CCONJ': 16}

lstm_pos = ForcedAlignmentCTCLSTM(1024, 1024, len(tag_to_ix.keys()))
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)
optimizer = torch.optim.Adam(lstm_pos.parameters(), lr=0.0001)

import pickle
with open('sample_data/X_train3.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('sample_data/X_test3.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('sample_data/y_train3.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('sample_data/y_test3.pkl', 'rb') as f:
    y_test = pickle.load(f)

# the all_data contains the the corresponding transcription, audio embedding and list of CTC tokens for each audio files
import pickle
with open('drive/MyDrive/all_data_train.pkl', 'rb') as f:
    all_data_train = pickle.load(f)

with open('drive/MyDrive/all_data_test.pkl', 'rb') as f:
    all_data_test = pickle.load(f)

all_train_ids = [a[2] for a in all_data_train]
all_test_ids = [a[2] for a in all_data_test]

train_word_segments = []
test_word_segments = []
c1 = 0
for a in all_train_ids:
  tups_train = create_tuples(a)
  train_word_segments.append(tups_train)


for a in all_test_ids:
  tups_test = create_tuples(a)
  test_word_segments.append(tups_test)

embeddings_train = [a[1][0] for a in all_data_train]
embeddings_test = [a[1][0] for a in all_data_test]

with open('drive/MyDrive/word_embeddings_train.pkl', 'rb') as f:
      data_train = pickle.load(f)

with open('drive/MyDrive/word_embeddings_test.pkl', 'rb') as f:
      data_test = pickle.load(f)

# SANITY CHECK

for i in range(len(data_train)):
  if len(data_train[i])!= len(all_data_train[i][0].split()):
    print(i)

for i in range(len(data_test)):
  if len(data_test[i])!= len(all_data_test[i][0].split()):
    print(i)

# NEWER APPROACH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.cuda import Device
from tqdm import tqdm
lstm_pos = lstm_pos.to(device)
ctc_loss = ctc_loss.to(device)
lstm_pos.train()
num_ = len(X_train)
num_test = len(X_test)

# NEWER APPROACH
def calculate_val_loss(model,data_test,y_test):
  count = 0
  total_loss = 0
  model.eval()
  with torch.no_grad():
    for b in tqdm(data_test):
      word_embeddings = b
      tags = prepare_sequence2(y_test[count][0],tag_to_ix)

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

#checkpoint = torch.load("drive/MyDrive/lstm_5_1/pos_tagger_words13.pt")
#lstm_pos.load_state_dict(checkpoint['model_state_dict'])
# model.load_states_dict(checkpoint['encoder_decoder_states_dict'])
# optimizer = torch.optim.Adam(lstm_pos.parameters(), lr=0.0001)
# WORDS : lstm_5_1
# dropout: 0.1 : 3 : 63,60
# dropout: 0.1 : 3 | Better than 0
# dropout: 0.5 : 3,5 : 3: 63,60 , 5: 64,58
# Check 1 and 5

# TRAINED WORDS : lstm_new_words
# dropout: 0 : 4 : 59,54
# dropout : 0.1 : 5 : 59,54
# dropout : 0.2 : 5 : 59,54
# dropout: 0.3 : 5 :  58,54
# dropout: 0.5 : 5 :  57,54

from tqdm import tqdm
train_losses = []
test_losses = []
for epoch in range(30):
    total_loss = 0
    count = 0
    lstm_pos.train()
    for b in tqdm(data_train):
      word_embeddings = b
      tags = prepare_sequence2(y_train[count][0],tag_to_ix)

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

      count +=1
    total_loss_test = calculate_val_loss(lstm_pos,data_test,y_test)
    print('Epoch',epoch)
    print('Train Loss ',total_loss/num_)
    print('Test Loss ',total_loss_test/num_test)

    train_losses.append(total_loss)
    test_losses.append(total_loss_test)


    torch.save({
            'model_state_dict': lstm_pos.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_list_train': train_losses,
            'loss_list_test': test_losses,
            'epoch': epoch,
            }, "drive/MyDrive/lstm_new_words/pos_tagger_new_words"+str(epoch)+".pt")

# EVALIUATION SEQUENCE
import torch.nn.functional as F
lstm_pos.eval()
from tqdm import tqdm
predicted_train = []
predicted_test = []
with torch.no_grad():
  for b in tqdm(data_train):
      wv = b
      wv = wv.to(device)
      tag_scores = lstm_pos(wv)
      _, indices = torch.max(tag_scores,1)
      ret = []
      for i in range(len(indices)):
          for key, value in tag_to_ix.items():
              if indices[i] == value:
                  ret.append((key))

      predicted_train.append(ret)

  for b in tqdm(data_test):
    wv = b
    wv = wv.to(device)
    tag_scores = lstm_pos(wv)
    _, indices = torch.max(tag_scores,1)
    ret = []
    for i in range(len(indices)):
        for key, value in tag_to_ix.items():
            if indices[i] == value:
                ret.append((key))

    predicted_test.append(ret)

ground_truth_train = []
ground_truth_test = []

for y in y_train:
  tags = y[0]
  temp = []
  for t in tags[1:len(tags)-1]:
    temp.append(t[0])

  ground_truth_train.append(temp)

for y in y_test:
  tags = y[0]
  temp = []
  for t in tags[1:len(tags)-1]:
    temp.append(t[0])

  ground_truth_test.append(temp)

predictions_train = []
actuals_train = []

predictions_test = []
actuals_test = []

for i in range(len(y_train)):
  t1 = y_train[i][1].split()
  actual = []
  for j in range(len(t1)):
    actual.append((t1[j],ground_truth_train[i][j]))
  actuals_train.append(actual)

for i in range(len(all_data_train)):
  t2 = all_data_train[i][0].split()
  prediction = []
  for j in range(len(t2)):
    prediction.append((t2[j],predicted_train[i][j]))
  predictions_train.append(prediction)

for i in range(len(y_test)):
  t1 = y_test[i][1].split()
  actual = []
  for j in range(len(t1)):
    actual.append((t1[j],ground_truth_test[i][j]))
  actuals_test.append(actual)

for i in range(len(all_data_test)):
  t2 = all_data_test[i][0].split()
  prediction = []
  for j in range(len(t2)):
    prediction.append((t2[j],predicted_test[i][j]))
  predictions_test.append(prediction)

ground_sentences_val = []

for y in y_test:
  ground_sentences_val.append(y[1])

from jiwer import wer
from jiwer import cer

def compute_metrics(pred_str,label_str):

    wer_ = wer(pred_str, label_str)
    #cer_ = cer(pred_str, label_str)

    return wer_

actual_sentence_val = [a[0] for a in all_data_test]

wers = []

for i in range(len(actual_sentence_val)):
  wers.append(compute_metrics(ground_sentences_val[i],actual_sentence_val[i]))

d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d6 = []
d7 = []
d8 = []
d9 = []
d10 = []
for i in range(len(wers)):
  if wers[i] >=0.0 and wers[i] <0.1:
    d1.append(i)

  elif wers[i] >=0.1 and wers[i] <0.2:
    d2.append(i)

  elif wers[i] >=0.2 and wers[i] <0.3:
    d3.append(i)

  elif wers[i] >=0.3 and wers[i] <0.4:
    d4.append(i)

  elif wers[i] >=0.4 and wers[i] <0.5:
    d5.append(i)

  elif wers[i] >=0.5 and wers[i] <0.6:
    d6.append(i)

  elif wers[i] >=0.6 and wers[i] <0.7:
    d7.append(i)

  elif wers[i] >=0.7 and wers[i] <0.8:
    d8.append(i)

  elif wers[i] >=0.8 and wers[i] <0.9:
    d9.append(i)

  elif wers[i] >=0.9 and wers[i] <=1.0:
    d10.append(i)


"""

d1 = []
d2 = []
d3 = []
d4 = []
d5 = []

for i in range(len(wers)):
  if wers[i] >=0.0 and wers[i] <0.2:
    d1.append(i)

  elif wers[i] >=0.2 and wers[i] <0.4:
    d2.append(i)

  elif wers[i] >=0.4 and wers[i] <0.6:
    d3.append(i)

  elif wers[i] >=0.6 and wers[i] <0.8:
    d4.append(i)

  else:
    d5.append(i)
"""

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

couples_train = []
for i in range(len(actuals_train)):
  if len(actuals_train[i]) == len(predictions_train[i]):
    couples_train.append((actuals_train[i],predictions_train[i]))
  else:
    gen1 = generate_lcs1(actuals_train[i],predictions_train[i])
    gen20 = generate_lcs2(predictions_train[i],gen1)
    gen21 = generate_lcs2(actuals_train[i],gen1)
    couples_train.append((gen20,gen21))

couples_test = []
for i in range(len(actuals_test)):
  if len(actuals_test[i]) == len(predictions_test[i]):
    couples_test.append((actuals_test[i],predictions_test[i]))
  else:
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

print(correct_train/total_train)

correct_test = 0
for tup in couples_test:
  l1 = tup[0]
  l2 = tup[1]

  for i in range(len(l1)):
    if l1[i][1] == l2[i][1]:
      correct_test += 1

print(correct_test/total_test)

def cal_pre_rec_f1(tag,data):
  tp = 0
  fn = 0
  fp = 0

  for tup in data:
    l1 = tup[0]
    l2 = tup[1]

    for i in range(len(l1)):

      if l1[i][1] == tag and l2[i][1] == tag:
        tp+=1

      elif l1[i][1]!= tag and l2[i][1] == tag:
        fn+=1

      elif l1[i][1] == tag and l2[i][1]!=tag:
        fp+=1

  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = (2*precision*recall)/(precision+recall)

  return {'Precision': precision, 'Recall': recall, 'F1_Score': f1_score}

def calc_f1(tag,indexes):
  tp = 0
  fn = 0
  fp = 0

  for i in indexes:
    l1 = couples_test[i][0]
    l2 = couples_test[i][1]

    for i in range(len(l1)):

      if l1[i][1] == tag and l2[i][1] == tag:
        tp+=1

      elif l1[i][1]!= tag and l2[i][1] == tag:
        fn+=1

      elif l1[i][1] == tag and l2[i][1]!=tag:
        fp+=1

  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = (2*precision*recall)/(precision+recall)

  return f1_score

keys = list(tag_to_ix.keys())
ignore = ['-']
datas = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10]
all_f1s = []
for d in datas:
  final_f1s = {}
  for k in keys:
    if k not in ignore:
      try:
        f1 = calc_f1(k,d)
        final_f1s[k] = f1

      except:
        continue

  all_f1s.append(final_f1s)

avg_f1s = []
for a in all_f1s:
  x = list(a.values())
  avg = sum(x)/len(x)
  avg_f1s.append(avg)

print(avg_f1s)

import matplotlib.pyplot as plt
x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.xlabel('WER')
plt.ylabel('F1-Score')
plt.plot(x,avg_f1s)

# F1 scores for all tags for complete dataset
keys = list(tag_to_ix.keys())
ignore = ['-']

final_scores = {}

for k in keys:
  if k not in ignore:
    try:
      scores = cal_pre_rec_f1(k)
      final_scores[k] = scores

    except:
      continue

print(final_scores)

print('Actual: ',actual)
print('Predicted: ',prediction)

