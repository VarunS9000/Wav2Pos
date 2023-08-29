import re
import pandas as pd
import torchaudio
import librosa
import numpy as np
import torchaudio
import re
import torch
import pickle
import random
from collections import defaultdict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from tqdm import tqdm

def remove_special_characters(sent):
    sent = re.sub(chars_to_ignore_regex, '', sent).lower()
    sent = re.sub(clean, '', sent)
    return sent

def speech_file_to_array_fn(audio):
    resampler = torchaudio.transforms.Resample(48_000, 16_000)
    return resampler(audio).squeeze().numpy()

def generate_speech_array(file_path):
  speech_array, sampling_rate = torchaudio.load(file_path)
  return speech_file_to_array_fn(speech_array)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clean = re.compile('<.*?>')
transcription = []
#chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\ \(\)\-]'
chars_to_ignore_regex = '[\,\?\.\!\-\;\"\“\%\‘\”\(\)\-\*]'

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

sentences_train = df1['transcription'].values
sentences_test = df2['transcription'].values

files_train = df1['file'].values
files_test = df2['file'].values

tokenizer = Wav2Vec2CTCTokenizer("vocab_asr.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2ForCTC.from_pretrained("checkpoint-2475")
model.config.output_hidden_states = True
model.config.output_attentions = True
model.to(device)

all_data_train = []
all_data_test = []

# The below loop shows the data being preprocessed and fed to the model one by one. For each audio, we pick out the predicted transcription, the audio embeddings
# and the ctc_token number for each of the predicted token. The ctc tokens helps me identify white spaces which is useful in developing corresponding word embeddings

for f in tqdm(files_train):
    g = generate_speech_array(f)
    speech, hidden_states, ctc_tokens = evaluate(g,processor,model)
    all_data_train.append(speech,hidden_states,ctc_tokens)

for f in tqdm(files_test):
    g = generate_speech_array(f)
    speech, hidden_states, ctc_tokens = evaluate(g,processor,model)
    all_data_test.append(speech,hidden_states,ctc_tokens)





