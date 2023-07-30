
from google.colab import drive
drive.mount('/content/drive')

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


import torch
import torchaudio
import re
import torch
import pickle
import random
from collections import defaultdict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer("sample_data/vocab_asr.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#processor.to(device)
model = Wav2Vec2ForCTC.from_pretrained("drive/MyDrive/checkpoint-2475")
model.config.output_hidden_states = True
model.config.output_attentions = True
model.to(device)
def evaluate(speech_array,processor,model):
    inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
    inputs.to(device)
    with torch.no_grad():
        m  = model(inputs.input_values, attention_mask=inputs.attention_mask)
        logits = m.logits
        hidden_states = m.hidden_states
    pred_ids = torch.argmax(logits, dim=-1)
    pred_speech = processor.batch_decode(pred_ids)

    return pred_speech[0], hidden_states[-1], pred_ids

import pickle
with open('sample_data/X_train3.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('sample_data/X_test3.pkl', 'rb') as f:
    X_test = pickle.load(f)

from tqdm import tqdm
all_data_train = []
files = X_train
for f in tqdm(files):
  s = generate_speech_array(f)
  a,b,c = evaluate(s,processor,model)
  all_data_train.append((a,b,c))

all_data_test = []
files = X_test
for f in tqdm(files):
  s = generate_speech_array(f)
  a,b,c = evaluate(s,processor,model)
  all_data_test.append((a,b,c))

with open('drive/MyDrive/all_data_train.pkl', 'wb') as f:
    pickle.dump(all_data_train,f)

with open('drive/MyDrive/all_data_test.pkl', 'wb') as f:
    pickle.dump(all_data_test,f)

import pickle
with open('drive/MyDrive/all_data_train.pkl', 'rb') as f:
    all_data_train = pickle.load(f)

with open('drive/MyDrive/all_data_test.pkl', 'rb') as f:
    all_data_test = pickle.load(f)

all_train_ids = [a[2] for a in all_data_train]
all_test_ids = [a[2] for a in all_data_test]

tups_train = create_tuples(all_train_ids[0])

train_word_segments = []
test_word_segments = []
c1 = 0
for a in all_train_ids:
  tups_train = create_tuples(a)
  train_word_segments.append(tups_train)


for a in all_test_ids:
  tups_test = create_tuples(a)
  test_word_segments.append(tups_test)

with open('drive/MyDrive/train_word_segments.pkl', 'wb') as f:
    pickle.dump(train_word_segments,f)

with open('drive/MyDrive/test_word_segments.pkl', 'wb') as f:
    pickle.dump(test_word_segments,f)

with open('drive/MyDrive/train_word_segments.pkl', 'rb') as f:
    train_word_segments = pickle.load(f)

with open('drive/MyDrive/test_word_segments.pkl', 'rb') as f:
    test_word_segments = pickle.load(f)

train_word_segments[0]

for i in range(len(train_word_segments)):
  if len(all_data_train[i][0].split())!=len(train_word_segments[i]):
    print(i,len(all_data_train[i][0].split()),len(train_word_segments[i]))

for i in range(len(test_word_segments)):
  if len(all_data_test[i][0].split())!=len(test_word_segments[i]):
    print(i,len(all_data_test[i][0].split()),len(test_word_segments[i]))

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

segments_for_encoder = []
for i in range(len(train_word_segments)):
  for j in range(len(train_word_segments[i])):
    t = train_word_segments[i][j]
    segments_for_encoder.append(all_data_train[i][1][0][t[0]:t[1]])

for i in range(len(test_word_segments)):
  for j in range(len(test_word_segments[i])):
    t = test_word_segments[i][j]
    segments_for_encoder.append(all_data_test[i][1][0][t[0]:t[1]])

test = []
for i in range(len(train_word_segments)):
  for j in range(len(train_word_segments[i])):
    t = train_word_segments[i][j]
    test.append(t)

for i in range(len(test_word_segments)):
  for j in range(len(test_word_segments[i])):
    t = test_word_segments[i][j]
    test.append(t)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMEncoderDecoder(1024, 1024,2)
#checkpoint = torch.load("drive/MyDrive/encoder_9.pt")
#model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

"""
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMEncoderDecoder(1024, 1024,2)
checkpoint = torch.load("drive/MyDrive/encoder_7.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
criterion = nn.MSELoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(8,11):
  total_loss = 0
  for s in tqdm(segments_for_encoder):
    out = model(s)
    loss = criterion(out,s)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  print('Epoch: ', epoch)
  print('Loss: ', total_loss/len(segments_for_encoder))

  torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "sample_data/encoder_"+str(epoch)+".pt")
"""

from tqdm import tqdm
word_vecs = []
with torch.no_grad():
  for i in tqdm(range(len(all_data_train))):
    sentence = []
    for j in range(len(train_word_segments[i])):
      out, (h,c) = model.encoder(all_data_train[i][1][0][train_word_segments[i][j][0]:train_word_segments[i][j][1]])
      sentence.append(h[0])
    word_vecs.append(torch.stack(sentence))

  with open('drive/MyDrive/word_embeddings_train.pkl', 'wb') as f:
      pickle.dump(word_vecs,f)

from tqdm import tqdm
word_vecs = []
with torch.no_grad():
  for i in tqdm(range(len(all_data_test))):
    sentence = []
    for j in range(len(test_word_segments[i])):
      out, (h,c) = model.encoder(all_data_test[i][1][0][test_word_segments[i][j][0]:test_word_segments[i][j][1]])
      sentence.append(h[0])
    word_vecs.append(torch.stack(sentence))

  with open('drive/MyDrive/word_embeddings_test.pkl', 'wb') as f:
      pickle.dump(word_vecs,f)
