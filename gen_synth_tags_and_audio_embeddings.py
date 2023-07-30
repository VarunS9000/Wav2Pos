

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from google.colab import drive
drive.mount('/content/drive')

import re
clean = re.compile('<.*?>')
transcription = []
#chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\ \(\)\-]'
chars_to_ignore_regex = '[\,\?\.\!\-\;\"\“\%\‘\”\(\)\-]'

def remove_special_characters(sent):
    sent = re.sub(chars_to_ignore_regex, '', sent).lower()
    sent = re.sub(clean, '', sent)
    return sent

import pandas as pd
df = pd.read_csv('drive/MyDrive/test.csv')

transcriptions = []
sentences = df['transcription'].values

for t in sentences:
  transcriptions.append(remove_special_characters(t))

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
Averaged perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""
from collections import defaultdict
import random
import pickle
class AveragedPerceptron(object):

	'''An averaged perceptron, as implemented by Matthew Honnibal.

	See more implementation details here:
		http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
	'''

	def __init__(self):
		# Each feature gets its own weight vector, so weights is a dict-of-dicts
		self.weights = {}
		self.classes = set()
		# The accumulated values, for the averaging. These will be keyed by
		# feature/clas tuples
		self._totals = defaultdict(int)
		# The last time the feature was changed, for the averaging. Also
		# keyed by feature/clas tuples
		# (tstamps is short for timestamps)
		self._tstamps = defaultdict(int)
		# Number of instances seen
		self.i = 0

	def predict(self, features):
		'''Dot-product the features and current weights and return the best label.'''
		scores = defaultdict(float)
		for feat, value in features.items():
			if feat not in self.weights or value == 0:
				continue
			weights = self.weights[feat]
			for label, weight in weights.items():
				scores[label] += value * weight
		# Do a secondary alphabetic sort, for stability
		return max(self.classes, key=lambda label: (scores[label], label))

	def update(self, truth, guess, features):
		'''Update the feature weights.'''
		def upd_feat(c, f, w, v):
			param = (f, c)
			self._totals[param] += (self.i - self._tstamps[param]) * w
			self._tstamps[param] = self.i
			self.weights[f][c] = w + v

		self.i += 1
		if truth == guess:
			return None
		for f in features:
			weights = self.weights.setdefault(f, {})
			upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
			upd_feat(guess, f, weights.get(guess, 0.0), -1.0)
		return None

	def average_weights(self):
		'''Average weights from all iterations.'''
		for feat, weights in self.weights.items():
			new_feat_weights = {}
			for clas, weight in weights.items():
				param = (feat, clas)
				total = self._totals[param]
				total += (self.i - self._tstamps[param]) * weight
				averaged = round(total / float(self.i), 3)
				if averaged:
					new_feat_weights[clas] = averaged
			self.weights[feat] = new_feat_weights
		return None

	def save(self, path):
		'''Save the pickled model weights.'''
		return pickle.dump(dict(self.weights), open(path, 'w'))

	def load(self, path):
		'''Load the pickled model weights.'''
		self.weights = pickle.load(open(path))
		return None


def train(nr_iter, examples):
	'''Return an averaged perceptron model trained on ``examples`` for
	``nr_iter`` iterations.
	'''
	model = AveragedPerceptron()
	for i in range(nr_iter):
		random.shuffle(examples)
		for features, class_ in examples:
			scores = model.predict(features)
			guess, score = max(scores.items(), key=lambda i: i[1])
			if guess != class_:
				model.update(class_, guess, features)
	model.average_weights()
	return model

#from transformers.utils.import_utils import SENTENCEPIECE_IMPORT_ERROR

def _pc(n, d):
	return (float(n) / d) * 100

class PerceptronTagger():

	'''Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
	See more implementation details here:
		http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
	:param load: Load the pickled model upon instantiation.
	'''

	START = ['-START-', '-START2-']
	END = ['-END-', '-END2-']

	def __init__(self, fname, load=True):
		self.model = AveragedPerceptron()
		self.tagdict = {}
		self.classes = set()
		self.model_file = fname
		if load:
			self.load(self.model_file)


	def tag(self, corpus, tokenise=False):
		punctuations = ['.',',','?',')','(',';']
		sentence = corpus
		for p in punctuations:
			sentence = sentence.replace(p,' '+p+' ')
		if 'yo:n ' in sentence:
			sentence = sentence.replace('yo:n ','yo n ')
		split_sent = sentence.split()
		for i in range(len(split_sent)):
			if ':' in split_sent[i] and len(split_sent[i])>1:
				temp = split_sent[i].split(':')
				sentence = sentence.replace(split_sent[i],temp[0]+temp[1])

		prev, prev2 = self.START
		tokens = []
		for words in [sentence.split()]:
				context = self.START + [self._normalise(w) for w in words] + self.END
				for i, word in enumerate(words):
						tag = self.tagdict.get(word)
						if not tag:
								features = self._get_features(i, word, context, prev, prev2)
								tag = self.model.predict(features)
						tokens.append((word, tag))
						prev2 = prev
						prev = tag
		return tokens







	def train(self, sentences, save_loc=None, nr_iter=5):
		'''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
		controls the number of Perceptron training iterations.
		:param sentences: A list of 10-value tuples
		:param save_loc: If not ``None``, saves a pickled model in this location.
		:param nr_iter: Number of training iterations.
		'''
		self._make_tagdict(sentences)
		self.model.classes = self.classes
		for iter_ in range(nr_iter):
			c = 0
			n = 0
#			for words,tags in sentences:
			for sentence in sentences:
#				print(c, n, '|||', sentence);
				print(n, end='')
				prev, prev2 = self.START
				context = self.START + [self._normalise(w[1]) for w in sentence] + self.END
				tags = [w[3] for w in sentence];
				for i, token in enumerate(sentence):
					word = token[1]
					guess = self.tagdict.get(word)
					if not guess:
						feats = self._get_features(i, word, context, prev, prev2)
						guess = self.model.predict(feats)
						self.model.update(tags[i], guess, feats)
					prev2 = prev
					prev = guess
					c += guess == tags[i]
					n += 1
				print('\r', end='')
			random.shuffle(sentences)
			print()
			print("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))
		self.model.average_weights()
		# Pickle as a binary file
		if save_loc is not None:
			pickle.dump((self.model.weights, self.tagdict, self.classes),
						 open(save_loc, 'wb'), -1)
		return None

	def load(self, loc):
		'''Load a pickled model.'''
		try:
			w_td_c = pickle.load(open(loc, 'rb'))
		except IOError:
			print("Missing " +loc+" file.")
		self.model.weights, self.tagdict, self.classes = w_td_c
		self.model.classes = self.classes
		return None

	def _normalise(self, word):
		'''Normalisation used in pre-processing.
		- All words are lower cased
		- Digits in the range 0000-2100 are represented as !YEAR;
		- Other digits are represented as !DIGITS
		:rtype: str
		'''
		if '-' in word and word[0] != '-':
			return '!HYPHEN'
		elif word.isdigit() and len(word) == 4:
			return '!YEAR'
		elif word[0].isdigit():
			return '!DIGITS'
		else:
			return word.lower()

	def _get_features(self, i, word, context, prev, prev2):
		'''Map tokens into a feature representation, implemented as a
		{hashable: float} dict. If the features change, a new model must be
		trained.
		'''
		def add(name, *args):
			features[' '.join((name,) + tuple(args))] += 1

		i += len(self.START)
		features = defaultdict(int)
		# It's useful to have a constant feature, which acts sort of like a prior
		add('bias')
		add('i suffix', word[-3:])
		add('i pref1', word[0])
		add('i-1 tag', prev)
		add('i-2 tag', prev2)
		add('i tag+i-2 tag', prev, prev2)
		add('i word', context[i])
		add('i-1 tag+i word', prev, context[i])
		add('i-1 word', context[i-1])
		add('i-1 suffix', context[i-1][-3:])
		add('i-2 word', context[i-2])
		add('i+1 word', context[i+1])
		add('i+1 suffix', context[i+1][-3:])
		add('i+2 word', context[i+2])
		#print(word, '|||', features)
		return features

	def _make_tagdict(self, sentences):
		'''Make a tag dictionary for single-tag words.'''
		counts = defaultdict(lambda: defaultdict(int))
#		for words, tags in sentences:
		for sentence in sentences:
			for token in sentence:
				word = token[1]
				tag = token[3]
				counts[word][tag] += 1
				self.classes.add(tag)
		freq_thresh = 20
		ambiguity_thresh = 0.97
		for word, tag_freqs in counts.items():
			tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
			n = sum(tag_freqs.values())
			# Don't add rare words to the tag dictionary
			# Only add quite unambiguous words
			if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
				self.tagdict[word] = tag

###############################################################################

def tagger(corpus_file, model_file):
	''' tag some text.
	:param corpus_file is a file handle
	:param model_file is a saved model file
	'''
	t = PerceptronTagger(model_file)
	tags = t.tag(corpus_file)
	return tags

def trainer(corpus_file, model_file):
	''' train a model
	:param corpus_file is a file handle
	:param model_file is a saved model file
	'''
	t = PerceptronTagger(model_file, load=False)
	sentences = [];
	for sent in corpus_file.read().split('\n\n'):
		sentence = []
		for token in sent.split('\n'):
			if token.strip() == '':
				continue
			if token[0] == '#':
				continue
			sentence.append(tuple(token.strip().split('\t')))
		sentences.append(sentence)

	t.train(sentences, save_loc=model_file, nr_iter=5)


f1 = open('sample_data/tree_bank.conllu')
trainer(f1,'sample_data/model.dat')

all_tags = []
for t in transcriptions:
  all_tags.append((tagger(t,'sample_data/model.dat'),t))

all_tags_final = []
for list_ in all_tags:
  temp = []
  for tup in list_[0]:
    temp.append((tup[1],len(tup[0])))
  all_tags_final.append((temp,list_[1]))


list_of_POS = []

for tags in all_tags_final:
  for t in tags[0]:
    list_of_POS.append(t[0])

list_of_POS = list(set(list_of_POS))

tag_to_ix = {}
for i in range(len(list_of_POS)):
  tag_to_ix[list_of_POS[i]] = i


from sklearn.model_selection import train_test_split
files = df['file'].values
X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2)

import pickle
with open('drive/MyDrive/lstm_audio/X_train3.pkl', 'wb') as f:
    pickle.dump(X_train,f)

with open('drive/MyDrive/lstm_audio/X_test3.pkl', 'wb') as f:
    pickle.dump(X_test,f)

with open('drive/MyDrive/lstm_audio/y_train3.pkl', 'wb') as f:
    pickle.dump(y_train,f)

with open('drive/MyDrive/lstm_audio/y_test.pkl3', 'wb') as f:
    pickle.dump(y_test,f)

