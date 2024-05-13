# Wav2Pos

This repository contains the implementation of the methods discussed in the paper titled *Wav2pos: Exploring syntactic analysis from audio for Highland Puebla Nahuatl* authored by Varun Sreedhar (myself), Robert Pugh, and Dr. Francis Tyers. The paper explores approaches to part-of-speech tagging from audio with minimal human-annotated data, focusing on Highland Puebla Nahuatl, a low-resource language of Mexico. Three different approaches are explained and implemented.

## Data Scraping and Channel Extraction

The `scraper_and_channel_extraction.py` in the `scraper_and_channel_extraction` folder contains code to scrape eaf files available in the openSLR repository index 92 to fetch the appropriate audio files with the correct channel and their corresponding transcriptions.

Data Description and Distribution

| Dataset               | Contents            | Sentences | Tokens |
|-----------------------|---------------------|-----------|--------|
| Wav2Vec2 train*       | OSLR - HPN          | 32k       | 285k   |
| Text POS Tagger train| (HPN-OSLR)∪WSPN     | 1.7k      | 17.6k  |
| Wav2pos train*        | OSLR - HPN          | 8k        | 71k    |
| Test data             | HPN ∩ OSLR          | 363       | 2.4k   |

**A description of the contents of the different datasets.** OSLR = OpenSLR data; HPN = Highland Puebla Nahuatl UD treebank; WSPN = Western Sierra Puebla Nahuatl UD Treebank. *The Wav2Vec2 fine-tuning data and the Wav2pos training data both come from the set of OpenSLR transcriptions not contained in the HPN treebank, but they are non-overlapping.*



## Wav2Vec2 Fine-Tuning

The `wav2vec.py` file contains code to fine-tune Wav2Vec2 for the ASR (automatic speech recognition) task on the Highland Puebla Nahuatl audio corpus. Wav2Vec2 is fine-tuned on 32k training data and validated on 8k validation data.

## Word-Based Approach

The word-based approach is implemented in `approach1_wb.py`, which describes a method where the audio embeddings and the word boundary information produced by the fine-tuned Wav2Vec2 are passed through an LSTM encoder to convert the audio embeddings to word embeddings. These embeddings are then passed through a Bi-LSTM network to perform the POS tagging. The corpus used for POS tagging operation is the 8k data points held out for validation in the Wav2Vec2 fine-tuning.

## Character-Based Approach

The character-based approach is implemented in `approach2_cb.py`, which describes a method where audio embeddings are tagged directly using a Bi-LSTM network, and the target labels are POS-tagged characters instead of words. For example, the phrase "kemah niyas", originally tagged [INTJ, VERB], would have the label sequence converted to [INTJ, INTJ, INTJ, INTJ, INTJ, SPACE, VERB, VERB, VERB, VERB, VERB]. A voting mechanism is used where the most frequently occurring tag in a word segment is assigned the appropriate tag for the word, using the word boundary information derived from fine-tuned Wav2Vec2.

## Averaged Perceptron Approach

The averaged perceptron approach involves training a perceptron network on a conllu file and then using it to tag the transcriptions produced by the fine-tuned Wav2Vec2 model. The script for training this perceptron can be found in the `train_ap_and_gen_synth_labels.py` file.

## Evaluation

The `evaluation.py` script evaluates all the above-mentioned approaches on 350+ unseen data. It's important to note that the ASR is not error-proof when it comes to predicting the number of words in a sentence. Therefore, during evaluation, the approaches are evaluated by selecting the correctly predicted words for each sentence and then performing a tag-to-tag accuracy test for those words.

## Results

apt - Averaged Perceptron Approach

wb - Word Based Approach

cb - Character Based Approach


| System | Accuracy | Precision | Recall | F1 | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|-----------|--------|----|
| apt    | 69.7     | 71.7      | 69.8   | 70.7 | 64.5      | 64.8   | 61.5 |
| wb     | 53.2     | 57.1      | 53.2   | 55.1 | 51.4      | 48.2   | 46.3 |
| cb     | 70.1     | 71.8      | 70.1   | 70.9 | 74.6      | 64.0   | 63.2 |

