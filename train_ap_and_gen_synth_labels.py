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

import pickle
f1 = open('Data/treebank_non_audio.conllu')
f2 = open('Data/nhi_itml-ud-test.conllu')

sents1 = f1.read().split('\n\n')
sents2 = f2.read().split('\n\n')

sents = sents1 + sents2

content = "\n\n".join(sents)

with open('Data/treebank_train.conllu', 'w') as file:
  file.write(content)

f3 = open('Data/treebank_train.conllu')

trainer(f3,'Data/model_non_audio_plus_nhi.dat')

train_transcriptions = []
test_transcriptions = []


import pickle

pickle_file_path1 = 'span_normalized/train_gold.pkl'
pickle_file_path2 = 'span_normalized/test_gold.pkl'

# Open the pickle file in read-binary mode
with open(pickle_file_path1, 'rb') as f:
    # Load the data from the pickle file
    train_transcriptions = pickle.load(f)

with open(pickle_file_path2, 'rb') as f:
    # Load the data from the pickle file
    test_transcriptions = pickle.load(f)

ap_predictions_train = []
ap_predictions_test = []

for t in train_transcriptions:
  ap_predictions_train.append(tagger(t,'Data/model_non_audio_plus_nhi.dat'))

for t in test_transcriptions:
  ap_predictions_test.append(tagger(t,'Data/model_non_audio_plus_nhi.dat'))

set_tags = set()

temp = ap_predictions_train + ap_predictions_test

for sent in temp:
  for tup in sent:
    set_tags.add(tup[1])

tag_to_ix = {'-':0,' ':1}

cnt = 2
for s in set_tags:
  tag_to_ix[s] = cnt
  cnt+=1
tag_to_ix

tag_to_ix0 = {'-':0}

cnt = 1
for s in set_tags:
  tag_to_ix0[s] = cnt
  cnt+=1

path1 = "Data/tag_to_ix0.pkl"
path2 = "Data/tag_to_ix.pkl"

with open(path1, 'wb') as f:
    # Load the data from the pickle file
    pickle.dump(tag_to_ix0,f)

with open(path2, 'wb') as f:
    # Load the data from the pickle file
    pickle.dump(tag_to_ix,f)

Y_train = []
Y_test = []

cnt = 0
for sent in ap_predictions_train:
  list_ = []
  for tup in sent:
    list_.append((tup[1],len(tup[0])))
  temp_tup = (list_,train_transcriptions[cnt])
  Y_train.append(temp_tup)
  cnt+=1

cnt = 0
for sent in ap_predictions_test:
  list_ = []
  for tup in sent:
    list_.append((tup[1],len(tup[0])))
  temp_tup = (list_,test_transcriptions[cnt])
  Y_test.append(temp_tup)
  cnt+=1

path1 = "span_normalized3/Y_train.pkl"
path2 = "span_normalized3/Y_test.pkl"

with open(path1, 'wb') as f:
    # Load the data from the pickle file
    pickle.dump(Y_train,f)

with open(path2, 'wb') as f:
    # Load the data from the pickle file
    pickle.dump(Y_test,f)
