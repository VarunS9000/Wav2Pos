from datasets import load_dataset, load_metric, Dataset
import re
import torch
import torchaudio
import nltk

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments

from transformers import Trainer
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_voice_train = load_dataset('csv', data_files='train_wav2vec.csv')

common_voice_test = load_dataset('csv', data_files='test_wav2vec.csv')


common_voice_train = common_voice_train['train']
common_voice_test = common_voice_test['train']

common_voice_train = common_voice_train.remove_columns(['channels'])
common_voice_test = common_voice_test.remove_columns(['channels'])


chars_to_ignore_regex = '[\,\?\.\!\-\;\"\“\%\‘\”\(\)\-]'
test = []
def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower() + " "
    return batch

tokenizer = Wav2Vec2CTCTokenizer("vocab_wav2vec.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["files"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["transcriptions"]
    return batch

common_voice_train = common_voice_train.map(speech_file_to_array_fn)
common_voice_test = common_voice_test.map(speech_file_to_array_fn)


import librosa
import numpy as np

def resample(batch):

  arr = np.asarray(batch["speech"])
  arr[np.isnan(arr)] = 0
  batch["speech"] = librosa.resample(arr, 48_000, 16_000)
  batch["sampling_rate"] = 16_000
  return batch

common_voice_train = common_voice_train.map(resample)
common_voice_test = common_voice_test.map(resample)

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=8, num_proc=4, batched=True)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)



@dataclass
class DataCollatorCTCWithPadding:
   
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True,
                                          pad_to_multiple_of=8, pad_to_multiple_of_labels=8)


def per(s1, s2):
    
    distance = nltk.edit_distance(s1, s2)

    # Calculate the WER
    per_ = float(distance) / max(len(s2),len(s1))

    return per_




def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    print(pred_str)
    print(label_str)

    n = len(pred_str)
    per_ = 0
    wer_ = 0
    for i in range(n):
        per_ += per(pred_str[i],label_str[i])
        wer_ += per(pred_str[i].split(),label_str[i].split())

    return {"per": per_/n, "wer": wer_/n}

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    mask_time_length=5,
    layerdrop=0.1,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)


model = model.to(device)
model.freeze_feature_extractor()
model.config.ctc_zero_infinity = True


training_args = TrainingArguments(
  output_dir='./wav2vec_nahuatl_check4',
  # output_dir="./wav2vec2-large-xlsr-nahuatl-demo",
  group_by_length=True,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=4,
  gradient_accumulation_steps=1,
  evaluation_strategy="steps",
  num_train_epochs= 100,
  fp16=True,
  save_steps=25,
  eval_steps=25,
  logging_steps=5,
  learning_rate=1e-4,
  warmup_steps=100,
  save_total_limit=1,
    
)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)


trainer.train()
