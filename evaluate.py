
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

# Averaged Perceptron

import pickle
import random
from collections import defaultdict
"""
Averaged perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""

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
		sentence = corpus
		if 'yo:n ' in sentence:
			sentence = sentence.replace('yo:n ','yo n ')
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
  cnt1 = 0
  cnt2 = 0
  for sent in corpus_file.read().split('\n\n'):

    sentence = []
    for token in sent.split('\n'):
      if token.strip() == '':
        continue
      if token[0] == '#':
        continue
      sentence.append(tuple(token.strip().split('\t')))
    sentences.append(sentence)
    cnt1+=1

  t.train(sentences, save_loc=model_file, nr_iter=5)

from conllu import parse
f = open('sample_data/tree_bank.conllu','r')
data = f.read()
sentence = parse(data)
print(sentence)



list_of_orig_sentence= []
list_of_metadata = []
sent_ids = []

for i in range(len(sentence)):
  if sentence[i].metadata['sent_id'].split(':')[0][-4:] == '.eaf':
    list_of_metadata.append(sentence[i].metadata)
    sent_ids.append(sentence[i].metadata['sent_id'])

print(sent_ids)

from conllu import parse_incr
from io import open

def extract_words_and_pos(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        #data = f.read()
        sentences = parse_incr(f)

        for sentence in sentences:
            words = [token["form"] for token in sentence]
            pos_tags = [token["upostag"] for token in sentence]
            yield words, pos_tags

# Replace 'data.conllu' with the path to your CoNLL-U file
file_path = "data/tree_bank.conllu"
count = 0
sent_and_pos = []
for sentence_words, sentence_pos in extract_words_and_pos(file_path):
    sent_and_pos.append((sentence_words,sentence_pos))
    count+=1


idx = 0
conllu_data = []
for s in sent_and_pos:
  a,b = s
  if sentence[idx].metadata['sent_id'] in sent_ids:
    conllu_data.append((a,b,sentence[idx].metadata))
  idx+=1

spans_dict = {'nikahsikamatisnekia': 'nikahsikamatis nekia', 'mawiltihtinemih': 'mawiltihti nemih', 'kijtosneki': 'kijtos neki', 'mochiujtiuetskej': 'mochiujti uetskej', 'okse': 'ok se', 'kalikampa': 'kali kampa', 'kuoujijtik': 'kuouj ijtik', 'talijtik': 'tal ijtik', 'semiuejueyi': 'semi uejueyi', 'kaltsintan': 'kal tsintan', 'tonalixkopa': 'tonalix kopa', 'kajfentenoj': 'kajfen tenoj', 'atpoliuikopa': 'atpoliui kopa', 'tamatisneki': 'tamatis neki', 'xaltepekopa': 'xaltepe kopa', 'kaltsintakal': 'kaltsinta kal', 'uantepekespaj': 'uan tepekespaj', 'atmolonkopa': 'atmolon kopa', 'tonalixkopatonalix': 'tonalixkopa tonalix', 'talixko': 'tal ixko', 'kaltsintaj': 'kal tsintaj', 'maseualkopa': 'maseual kopa', 'koyokopakoyo': 'koyokopa koyo', 'majase': 'maja se', 'ojtenoj': 'oj tenoj', 'kuoujtajpa': 'kuoujtaj pa', 'kaltenoj': 'kal tenoj', 'kuouijtik': 'kuou ijtik', 'chilartenoj': 'chilar tenoj', 'tamixochiyoua': 'tami xochiyoua', 'koyokopa': 'koyo kopa', 'kikuasneki': 'kikuas neki', 'maajsi': 'ma ajsi', 'xolalpan': 'xolal pan', 'inixiujyo': 'in ixiujyo', 'miltenojmil': 'miltenoj mil', 'tatampakopatatampa': 'tatampakopa tatampa', 'nexaltipampa': 'ne xaltipampa', 'tatampa': 'ta tampa', 'aten': 'a ten', 'atentenoha': 'atentenoh a', 'imaikan': 'imai kan', 'kitokasneki': 'kitokas neki', 'imatampa': 'ima tampa', 'imatampaima': 'imatampa ima', 'nikiitaseki': 'nikiita seki', 'ehekaixko': 'eheka ixko', 'kanitehwatsin': 'kani tehwatsin'}

def remove_punct_and_spans(data):
  new_data = []
  ground_eval_sentences = []
  for d in data:
    tokens = d[0]
    tags = d[1]
    metadata = d[2]

    token_and_tag = []
    new_tokens= []
    for i in range(len(tokens)):
      if tokens[i].lower() not in spans_dict and tags[i]!='PUNCT':
        token_and_tag.append((tokens[i].lower(),tags[i]))
        new_tokens.append(tokens[i].lower())


    sent = " ".join(new_tokens)
    new_data.append((sent,token_and_tag,metadata))

  return new_data

data = remove_punct_and_spans(conllu_data)

eval_gold = [d[0] for d in data]
actual_eval = [d[1] for d in data]

with open('data/final_eval_data.pkl', 'wb') as f:
    pickle.dump(data,f)

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

def normalize(str_):
  norm_dict = {
   'â':'a',
   'ã':'a',
   'á':'a',
   'é':'e',
   'í':'i',
   'ó':'o',
   'ú':'u',
   'ñ':'n'
  }

  str_ = str_.replace('¿','')
  str_ = str_.replace('[','')
  str_ = str_.replace(']','')
  str_ = str_.replace('©','')
  str_ = str_.replace('`','')
  str_ = str_.replace('³','')
  str_ = str_.replace('º','')
  str_ = str_.replace('¨','')

  for k in norm_dict:
    str_ = str_.replace(k,norm_dict[k])

  return str_

import torch
import torchaudio
import re
import torch
import pickle
import random
from collections import defaultdict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer("data/vocab_wav2vec.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#processor.to(device)
model = Wav2Vec2ForCTC.from_pretrained("Wav2Vec2/checkpoint-7175")
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
    pred_speech = processor.batch_decode(pred_ids, output_word_offsets=True)
    list_ = [d['word'] for d in pred_speech.word_offsets[0]]
    string_ = " ".join(list_)
    string_ = normalize(string_)
    #tups = [(d['start_offset'],d['end_offset']) for d in pred_speech.word_offsets[0]]


    return string_, hidden_states[-1], pred_speech.word_offsets

with open('data/final_eval_data.pkl','rb') as f:
    final_eval_data = pickle.load(f)
    
eval_data = []
cnt = 0

actual_eval = [e[-1] for e in final_eval_data]


from tqdm import tqdm
eval_inner_data = []
predictions = []
root_path = 'AudioData/'
for f in tqdm(eval_data):
  audio = root_path + f[1]
  g = generate_speech_array(audio)
  a,b,c = evaluate(g,processor,model)
  eval_inner_data.append((a,b,c))
  predictions.append(a)

with open('data/eval_inner_data.pkl','wb') as f:
    pickle.dump(eval_inner_data,f)

with open('data/eval_inner_data.pkl','rb') as f:
    eval_inner_data = pickle.load(f)
    
predictions = [e[0] for e in eval_inner_data]

with open('data/tag_to_ix.pkl','rb') as f:
    tag_to_ix = pickle.load(f)

with open('data/tag_to_ix0.pkl','rb') as f:
    tag_to_ix0 = pickle.load(f)

import pickle
spans_dict = {'nikahsikamatisnekia': 'nikahsikamatis nekia', 'mawiltihtinemih': 'mawiltihti nemih', 'kijtosneki': 'kijtos neki', 'mochiujtiuetskej': 'mochiujti uetskej', 'okse': 'ok se', 'kalikampa': 'kali kampa', 'kuoujijtik': 'kuouj ijtik', 'talijtik': 'tal ijtik', 'semiuejueyi': 'semi uejueyi', 'kaltsintan': 'kal tsintan', 'tonalixkopa': 'tonalix kopa', 'kajfentenoj': 'kajfen tenoj', 'atpoliuikopa': 'atpoliui kopa', 'tamatisneki': 'tamatis neki', 'xaltepekopa': 'xaltepe kopa', 'kaltsintakal': 'kaltsinta kal', 'uantepekespaj': 'uan tepekespaj', 'atmolonkopa': 'atmolon kopa', 'tonalixkopatonalix': 'tonalixkopa tonalix', 'talixko': 'tal ixko', 'kaltsintaj': 'kal tsintaj', 'maseualkopa': 'maseual kopa', 'koyokopakoyo': 'koyokopa koyo', 'majase': 'maja se', 'ojtenoj': 'oj tenoj', 'kuoujtajpa': 'kuoujtaj pa', 'kaltenoj': 'kal tenoj', 'kuouijtik': 'kuou ijtik', 'chilartenoj': 'chilar tenoj', 'tamixochiyoua': 'tami xochiyoua', 'koyokopa': 'koyo kopa', 'kikuasneki': 'kikuas neki', 'maajsi': 'ma ajsi', 'xolalpan': 'xolal pan', 'inixiujyo': 'in ixiujyo', 'miltenojmil': 'miltenoj mil', 'tatampakopatatampa': 'tatampakopa tatampa', 'nexaltipampa': 'ne xaltipampa', 'tatampa': 'ta tampa', 'aten': 'a ten', 'atentenoha': 'atentenoh a', 'imaikan': 'imai kan', 'kitokasneki': 'kitokas neki', 'imatampa': 'ima tampa', 'imatampaima': 'imatampa ima', 'nikiitaseki': 'nikiita seki', 'ehekaixko': 'eheka ixko', 'kanitehwatsin': 'kani tehwatsin'}
def create_tuples(data_list):
  data_list = data_list[0]
  tups = []
  start = 0
  for d in data_list:
    tups.append((start,d['end_offset']))
    start = d['end_offset'] + 1

  return tups

def span_normalize(str_):
  for k in spans_dict:
    str_ = str_.replace(k,spans_dict[k])

  return str_




data_list_eval = [e[-1] for e in eval_inner_data]

init_tuples_eval = [create_tuples(d) for d in data_list_eval]


final_tuples_eval = []

predictions_eval = []

for i in range(len(predictions)):

  words = []
  temp = []
  ref_words = predictions[i].split()
  for j in range(len(ref_words)):
    t = init_tuples_eval[i][j]
    if ref_words[j] in spans_dict:
      word = spans_dict[ref_words[j]].split()
      avg = (t[0]+t[1])//2
      t1 = (t[0],avg)
      t2 = (avg+1,t[1])
      temp.append(t1)
      temp.append(t2)


      for w in word:
        words.append(w)

    else:
      words.append(ref_words[j])
      temp.append(t)

  final_tuples_eval.append(temp)

  predictions_eval.append(" ".join(words))

ap_tags_on_eval = []
for p in predictions_eval:
  ap_tags_on_eval.append(tagger(p,'data/model_non_audio_plus_nhi.dat'))

tags = []
for list_ in ap_tags_on_eval:
  for tup in list_:
    if tup[1] not in tags:
      tags.append(tup[1])

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



X_eval = []
count = 0
with torch.no_grad():
  for e in eval_inner_data:
    sentence = []
    b = e[1][0]
    for j in range(len(final_tuples_eval[count])):
      out, (h,c) = model1.encoder(b[final_tuples_eval[count][j][0]:final_tuples_eval[count][j][1]+1])
      sentence.append(h[0])
    X_eval.append(torch.stack(sentence))
    count+=1

from tqdm import tqdm
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


from collections import Counter

def top_occurrences(input_list):
    # Count occurrences of each element in the list
    element_counts = Counter(input_list)

    # Get the top 3 occurring elements and their counts
    top_elements = element_counts.most_common(3)

    return top_elements

def get_accuracy2(sent_predictions):
  predictions_ev = []

  c = 0
  with torch.no_grad():
    for b in X_eval:


        word_embeddings = b

        input_length = torch.tensor(len(word_embeddings))

        input_length = input_length.to(device)

        word_embeddings = word_embeddings.to(device)

        output = lstm_pos(word_embeddings)
        _, indices = torch.max(output,1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_ix0.items():
                if indices[i] == value:
                    ret.append(key)
        temp = []
        x = sent_predictions[c].split()
        for i in range(len(b)):
          temp.append((x[i],ret[i]))

        predictions_ev.append(temp)

        c+=1





  couples_eval = []
  for i in range(len(actual_eval)):
    gen1 = generate_lcs1(actual_eval[i],predictions_ev[i])
    gen20 = generate_lcs2(predictions_ev[i],gen1)
    gen21 = generate_lcs2(actual_eval[i],gen1)
    couples_eval.append((gen20,gen21))


  total_eval = 0
  for c in couples_eval:
    total_eval+=len(c[0])

  correct_eval = 0
  for tup in couples_eval:
    l1 = tup[0]
    l2 = tup[1]

    for i in range(len(l1)):
      if l1[i][1] == l2[i][1]:
        correct_eval += 1

  eval_accuracy = correct_eval/total_eval

  print('Eval Accuracy: ',eval_accuracy)
  return couples_eval

def get_accuracy(sent_predictions):

  predictions_ev = []

  c = 0
  print('Fetching eval results')
  with torch.no_grad():
    for b in tqdm(eval_inner_data):

      embeddings = b[1][0]

      embeddings = embeddings.to(device)

      output = lstm_pos(embeddings)
      _, indices = torch.max(output,1)
      ret = []
      for i in range(len(indices)):
          for key, value in tag_to_ix.items():
              if indices[i] == value:
                  ret.append(key)

      new_ret_train = []

      for tup in final_tuples_eval[c]:
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
      x = sent_predictions[c].split()
      for i in range(len(new_ret_train)):
        temp.append((x[i],new_ret_train[i]))

      predictions_ev.append(temp)



      c+=1



  print('Generating eval subsequences')
  couples_eval = []
  for i in tqdm(range(len(actual_eval))):

    gen1 = generate_lcs1(actual_eval[i],predictions_ev[i])
    gen20 = generate_lcs2(predictions_ev[i],gen1)
    gen21 = generate_lcs2(actual_eval[i],gen1)
    couples_eval.append((gen20,gen21))



  total_eval = 0
  for c in couples_eval:
    total_eval+=len(c[0])

  correct_eval = 0
  for tup in couples_eval:
    l1 = tup[0]
    l2 = tup[1]

    for i in range(len(l1)):
      if l1[i][1] == l2[i][1]:
        correct_eval += 1

  eval_accuracy = correct_eval/total_eval
  print('Eval Accuracy: ',eval_accuracy)
  return couples_eval

couples_eval_ap = []
for i in range(len(actual_eval)):
  gen1 = generate_lcs1(actual_eval[i],ap_tags_on_eval[i])
  gen20 = generate_lcs2(ap_tags_on_eval[i],gen1)
  gen21 = generate_lcs2(actual_eval[i],gen1)
  couples_eval_ap.append((gen20,gen21))


total_eval = 0
for c in couples_eval_ap:
  total_eval+=len(c[0])

correct_eval = 0
for tup in couples_eval_ap:
  l1 = tup[0]
  l2 = tup[1]

  for i in range(len(l1)):
    if l1[i][1] == l2[i][1]:
      correct_eval += 1

eval_accuracy = correct_eval/total_eval

print('Eval Accuracy: ',eval_accuracy)

with open('data/avg_perceptron_results_on_predicted_eval_trancript.pkl', 'wb') as file:
    # Dump the data into the file using pickle
    pickle.dump(couples_eval_ap, file)

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

all_couples = []

for i in range(10):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(f"drive/MyDrive/Part1/lstm_pos_{i}.pt")
  lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix0.keys()),0.3)
  lstm_pos.load_state_dict(checkpoint['model_state_dict'])
  lstm_pos = lstm_pos.to(device)
  lstm_pos.eval()
  print(f"Epoch {i+1}")
  couples1 = get_accuracy2(predictions_eval)
  all_couples.append(couples1)

with open('sample_data/approach1_results.pkl', 'wb') as file:
    # Dump the data into the file using pickle
    pickle.dump(all_couples[-1], file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(f"drive/MyDrive/span_normalized3/Part2/lstm_pos_27.pt")
lstm_pos = ForcedAlignmentCTCLSTM(1024, 512, len(tag_to_ix.keys()),0.2)
lstm_pos.load_state_dict(checkpoint['model_state_dict'])
lstm_pos = lstm_pos.to(device)
lstm_pos.eval()
print(f'Fetching accuracies for epoch 30')
couples2= get_accuracy(predictions_eval)
#print(f'Eval Accuracy: {eval_acc}')

with open('data/approach2_results.pkl', 'wb') as file:
    # Dump the data into the file using pickle
    pickle.dump(couples2, file)

# 95.47 84.75 83.52
def calc_f1(tag, couples):
  tp = 0
  fn = 0
  fp = 0

  l1 = couples[0]
  l2 = couples[1]

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

  return precision, recall, f1_score

def cal_pre_rec_f1(tag,data):
  tp = 0
  fn = 0
  fp = 0

  for tup in data:
    l1 = tup[0]
    l2 = tup[1]
    if len(l1)>0:
      for i in range(len(l1)):

        if l1[i][1] == tag and l2[i][1] == tag:
          tp+=1

        elif l1[i][1]!= tag and l2[i][1] == tag:
          fn+=1

        elif l1[i][1] == tag and l2[i][1]!=tag:
          fp+=1

  d1 = tp+fp
  d2 = tp+fn
  if d1 == 0 or d2 == 0:
    return

  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  if precision + recall == 0:
    return
  f1_score = (2*precision*recall)/(precision+recall)



  return {'Precision': precision, 'Recall': recall, 'F1_Score': f1_score}


def cal_confusion_matrix(tag,data):
  tp = 0
  tn = 0
  fn = 0
  fp = 0

  for tup in data:
    l1 = tup[0]
    l2 = tup[1]
    if len(l1)>0:
      for i in range(len(l1)):

        if l1[i][1] == tag and l2[i][1] == tag:
          tp+=1

        elif l1[i][1] != tag and l2[i][1] != tag:
          tn+=1

        elif l1[i][1]!= tag and l2[i][1] == tag:
          fn+=1

        elif l1[i][1] == tag and l2[i][1]!=tag:
          fp+=1

  return {'True Positives': tp, 'True Negetives': tn, 'False Positives': fp, 'False Negetives': fn}



couples1 = all_couples[-1]
metrics1 = {}
tags11 = []
for k in tag_to_ix0:
  if k not in ['-',' ']:
    tags11.append(k)
for k in tags11:
  metrics1[k] = cal_pre_rec_f1(k,couples1)



confusion1 = {}
tags12 = []
for k in tag_to_ix0:
  if k not in ['-',' ']:
    tags12.append(k)

for k in tags12:
  confusion1[k] = cal_confusion_matrix(k,couples1)




metrics2 = {}
tags21 = []
for k in tag_to_ix:
  if k not in ['-',' ']:
    tags21.append(k)
for k in tags21:
  metrics2[k] = cal_pre_rec_f1(k,couples2)



confusion2 = {}
tags22 = []
for k in tag_to_ix:
  if k not in ['-',' ']:
    tags22.append(k)

for k in tags22:
  confusion2[k] = cal_confusion_matrix(k,couples2)


metrics3 = {}
confusion3 = {}
for k in tags:
  metrics3[k] = cal_pre_rec_f1(k,couples_eval_ap)

for k in tags:
  confusion3[k] = cal_confusion_matrix(k,couples_eval_ap)