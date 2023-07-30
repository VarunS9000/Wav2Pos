
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(torch.__version__)

from google.colab import drive
drive.mount('/content/drive')

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

def preprocess_target(tags):
  y = []
  for t in tags:
    n = 1
    if t[1]>1:
      n = t[1]//2
    for i in range(n):
      if i<n - 1:
        y.append(t[0])
      else:
        y.append(t[0])
        y.append('@')
        y.append('@')
  y.pop()
  y.pop()
  return y

def preprocess_target2(tags):

  y = []
  for t in tags:
    y.append(t[0])
    y.append(" ")

  y.pop()

  return y
def prepare_sequence(tags,i2x):
  seq = []
  for t in tags:
    seq.append(i2x[t])

def prepare_sequence2(tags,i2x):
  seq = []
  n = len(tags)
  for t in tags[1:n-1]:
    seq.append(i2x[t[0]])


  return torch.tensor(seq, dtype=torch.long)

"""
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

    return pred_speech[0], hidden_states[-1]

"""

from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ForcedAlignmentCTCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size1, output_size2):
        super(ForcedAlignmentCTCLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers = 3, bidirectional = True, batch_first=True, dropout = 0.5)
        self.fc1 = nn.Linear(hidden_size*2, output_size1)
        self.fc2 = nn.Linear(hidden_size*2, output_size2)



    def forward(self, x):
        x, _ = self.lstm(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x1 = log_softmax(x1, dim=-1)
        x2 = log_softmax(x2, dim=-1)
        return x1,x2

tag_to_ix = {'-': 0, " ": 1, 'ADV': 2, 'NOUN': 3, 'AUX': 4, 'ADP': 5, '_': 6, 'SCONJ': 7, 'VERB': 8, 'PROPN': 9, 'PUNCT': 10, 'PRON': 11, '<S>': 12, 'X': 13, 'ADJ': 14, '<E>': 15, 'NUM': 16, 'INTJ': 17, 'DET': 18, 'CCONJ': 19}
#tag_to_ix = {'-': 0, 'ADV': 1, 'NOUN': 2, 'AUX': 3, 'ADP': 4, '_': 5, 'SCONJ': 6, 'VERB': 7, 'PROPN': 8, 'PUNCT': 9, 'PRON': 10, 'X': 11, 'ADJ': 12, 'NUM': 13, 'INTJ': 14, 'DET': 15, 'CCONJ': 16}
#tag_to_ix = {'-': 0, '@': 1, 'ADV': 2, 'NOUN': 3, 'AUX': 4, 'ADP': 5, '_': 6, 'SCONJ': 7, 'VERB': 8, 'PROPN': 9, 'PUNCT': 10, 'PRON': 11, '<S>': 12, 'X': 13, 'ADJ': 14, '<E>': 15, 'NUM': 16, 'INTJ': 17, 'DET': 18, 'CCONJ': 19}


import json
with open('sample_data/vocab_lstm.json', 'r') as vocab_file:
    vocab_dict = json.load(vocab_file)

lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix.keys()), len(vocab_dict.keys()))
ctc_loss1 = nn.CTCLoss(blank=0,zero_infinity=True)
ctc_loss2 = nn.CTCLoss(blank=59,zero_infinity=True)
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

import pickle
with open('drive/MyDrive/lstm/embeddings_train3.pkl', 'rb') as f:
    embeddings_train = pickle.load(f)

with open('drive/MyDrive/lstm/embeddings_test3.pkl', 'rb') as f:
    embeddings_test = pickle.load(f)

from torch.cuda import Device
from tqdm import tqdm
lstm_pos = lstm_pos.to(device)
ctc_loss1 = ctc_loss1.to(device)
ctc_loss2 = ctc_loss2.to(device)
lstm_pos.train()
num_ = len(X_train)
num_test = len(X_test)
print(len(y_train))
print(len(embeddings_train))

def calculate_val_loss(model,embeddings_test,y_test):
  count = 0
  total_loss1 = 0
  total_loss2 = 0
  for b in tqdm(embeddings_test):
    audio_features = b
    tags = prepare_sequence(preprocess_target2(y_test[count][0]),tag_to_ix)
    transcription = prepare_sequence(y_test[count][1],vocab_dict)

    input_length1 = torch.tensor(len(audio_features))
    target_length1 = torch.tensor(len(tags))

    input_length1 = input_length1.to(device)
    target_length1 = target_length1.to(device)

    input_length2 = torch.tensor(len(audio_features))
    target_length2 = torch.tensor(len(transcription))

    input_length2 = input_length2.to(device)
    target_length2 = target_length2.to(device)

    audio_features = audio_features.to(device)
    tags = tags.to(device)

    outputs1,outputs2 = lstm_pos(audio_features)

    loss1 = ctc_loss1(outputs1, tags,input_length1,target_length1)
    loss2 = ctc_loss2(outputs2, transcription,input_length2,target_length2)

    total_loss1 += loss1.item()
    total_loss2 += loss2.item()

    count +=1

  return total_loss1, total_loss2

# 20 - 0.4
# 7,8,9 lstm5
# 7 : 84.74 , 72.4
# 8 : 86.82 , 72.4
# 5 : 80.2 ,  72.5
# 6 : 82.6 ,  72.7

from tqdm import tqdm
train_losses = []
test_losses = []
for epoch in range(30):
    total_loss1 = 0
    total_loss2 = 0
    count = 0
    for b in tqdm(embeddings_train):
        audio_features = b
        tags = prepare_sequence(preprocess_target2(y_train[count][0]),tag_to_ix)
        transcription = prepare_sequence(y_train[count][1],vocab_dict)

        input_length1 = torch.tensor(len(audio_features))
        target_length1 = torch.tensor(len(tags))

        input_length1 = input_length1.to(device)
        target_length1 = target_length1.to(device)

        input_length2 = torch.tensor(len(audio_features))
        target_length2 = torch.tensor(len(transcription))

        input_length2 = input_length2.to(device)
        target_length2 = target_length2.to(device)

        audio_features = audio_features.to(device)
        tags = tags.to(device)

        outputs1,outputs2 = lstm_pos(audio_features)

        loss1 = ctc_loss1(outputs1, tags,input_length1,target_length1)
        loss2 = ctc_loss2(outputs2, transcription,input_length2,target_length2)


        optimizer.zero_grad()
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()


        count+=1

    total_loss_test = calculate_val_loss(lstm_pos,embeddings_test,y_test)
    print('Epoch',epoch)
    print('Train Loss Tagger',total_loss1/num_)
    print('Train Loss ASR',total_loss2/num_)
    print('Validation Loss Tagger',total_loss_test[0]/num_test)
    print('Validation Loss ASR',total_loss_test[1]/num_test)
    print('Train Loss Both', (total_loss1 + total_loss2)/num_)
    print('Test Loss Both', (total_loss_test[0] + total_loss_test[1])/num_test)
    train_losses.append(total_loss1 + total_loss2)
    test_losses.append(total_loss_test[0] + total_loss_test[1])


    torch.save({
            'model_state_dict': lstm_pos.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_list_train': train_losses,
            'loss_list_test': test_losses,
            'epoch': epoch,
            }, "drive/MyDrive/lstm_5_1/audio_pos_tagger_checkpoint_"+str(epoch)+".pt")

def find_matching_substring(s, input_str):
    n = len(s)
    count = 0
    matching_substring = 'NOPE'
    for i in range(n):
        for j in range(i+1, n+1):
            if s[i:j] == input_str:
                count += 1
                matching_substring = s[i:j]
    return matching_substring, count

def avoid_repition(str_):
  stack = ['-']

  for s in str_:
    if stack[-1]!= s:
      stack.append(s)

  return "".join(stack)

lstm_pos.eval()
from tqdm import tqdm
predicted_sent_train  = []
predicted_sent_test  = []
with torch.no_grad():
    for b in tqdm(embeddings_train):

        audio_features = b
        audio_features = audio_features.to(device)
        tag_scores, asr_scores  = lstm_pos(audio_features)
        asr_scores = asr_scores.to(device)
        _, indices = torch.max(asr_scores,1)
        ret = []
        for i in range(len(indices)):
            for key, value in vocab_dict.items():
                if indices[i] == value:
                    ret.append((key))

        x = "".join(ret)
        x = avoid_repition(x)
        x = x.replace('[UNK]','')
        x = x.split()
        x = " ".join(x)

        predicted_sent_train.append(x[1:])


with torch.no_grad():
    for b in tqdm(embeddings_test):

        audio_features = b
        audio_features = audio_features.to(device)
        tag_scores, asr_scores  = lstm_pos(audio_features)
        asr_scores = asr_scores.to(device)
        _, indices = torch.max(asr_scores,1)
        ret = []
        for i in range(len(indices)):
            for key, value in vocab_dict.items():
                if indices[i] == value:
                    ret.append((key))

        x = "".join(ret)
        x = avoid_repition(x)
        x = x.replace('[UNK]','')
        x = x.split()
        x = " ".join(x)

        predicted_sent_test.append(x[1:])

#compute_metrics(predicted_tags_train[0],tags_train[0],predicted_tags_test[0],tags_test[0])

def similarity(s1,s2):

  distance = torchaudio.functional.edit_distance(s1, s2)

  # max length of the two strings
  length = max(len(s1), len(s2))

  #calculate the percentage of similarity
  similarity_score = 1 - (distance/length)

  return similarity_score*100

train_sent - [y[1] for y in y_train]
test_sent - [y[1] for y in y_test]

sim_sent_train = []
sim_sent_test = []

for i in range(len(train_sent)):
  sim_sent_train.append(similarity(train_sent[i],predicted_sent_train[i]))

print('Train Accuracy',sum(sim_sent_train)/len(sim_sent_train))

for i in range(len(test_sent)):
  sim_sent_test.append(similarity(test_sent[i],predicted_sent_test[i]))

print('Test Accuracy',sum(sim_sent_test)/len(sim_sent_test))

from tqdm import tqdm
predicted_tags_train = []
predicted_tags_test  = []
lstm_pos.eval()
with torch.no_grad():
    for b in tqdm(embeddings_train):
        #s = generate_speech_array(f)
        #a,b = evaluate(s,processor,model)
        audio_features = b
        audio_features = audio_features.to(device)
        tag_scores, asr_scores = lstm_pos(audio_features)
        tag_scores = tag_scores.to(device)
        _, indices = torch.max(tag_scores,1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_ix.items():
                if indices[i] == value:
                    ret.append((key))
        x1 = "".join(ret)
        x1 = x1.replace('-',' ')

        x1 = x1.split()

        for i in range(len(x1)):
          if x1[i] not in tag_to_ix:

            elements = []
            for t in tag_to_ix:
              elements.append(find_matching_substring(x1[i],t))

            sorted_ele = sorted(elements, key=lambda x: x[1])
            x1[i] = sorted_ele[-1][0]


        predicted_tags_train.append(x1)


with torch.no_grad():
    for b in tqdm(embeddings_test):
        #s = generate_speech_array(f)
        #a,b = evaluate(s,processor,model)
        audio_features = b
        audio_features = audio_features.to(device)
        tag_scores, asr_scores = lstm_pos(audio_features)
        tag_scores = tag_scores.to(device)
        _, indices = torch.max(tag_scores,1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_ix.items():
                if indices[i] == value:
                    ret.append((key))
        x1 = "".join(ret)
        x1 = x1.replace('-',' ')

        x1 = x1.split()

        for i in range(len(x1)):
          if x1[i] not in tag_to_ix:

            elements = []
            for t in tag_to_ix:
              elements.append(find_matching_substring(x1[i],t))

            sorted_ele = sorted(elements, key=lambda x: x[1])
            x1[i] = sorted_ele[-1][0]


        predicted_tags_test.append(x1)

tags_train = []
for i in range(len(y_train)):
  temp = []
  for j in range(len(y_train[i][0])):
    temp.append(y_test[i][0][j][0])
  tags_train.append(temp)

tags_test = []
for i in range(len(y_test)):
  temp = []
  for j in range(len(y_test[i][0])):
    temp.append(y_test[i][0][j][0])
  tags_test.append(temp)

sim_train = []
for i in range(len(tags_train)):
  sim_train.append(similarity(tags_train[i][1:len(tags_train[i])-1],predicted_tags_test[i][1:len(predicted_tags_train[i])-1]))

print('Train Accuracy',sum(sim_train)/len(sim_train))

sim_test = []
for i in range(len(tags_test)):
  sim_test.append(similarity(tags_test[i][1:len(tags_test[i])-1],predicted_tags_test[i][1:len(predicted_tags_test[i])-1]))

print('Test Accuracy',sum(sim_test)/len(sim_test))

