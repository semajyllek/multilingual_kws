from typing import List, NamedTuple

from datasets import concatenate_datasets, load_dataset, Dataset
from collections import defaultdict
from pathlib import Path
import tensorflow as tf
import numpy as np
import json

import IPython



SEARCH_SPLITS = ['train', 'test', 'dev']
FEW = 5


class AssetPack:
    def __init__(self) -> None:
        self.background_noise_path: str = None
        self.unknown_files: List[str] = None

    def __repr__(self) -> str:
        return f"background_noise_path: {self.background_noise_path}, unknown_files: {self.unknown_files}"


  





def pathfix_unknown_files(unknown_txt_path: Path) -> List[str]:
    unknown_files=[]
    with open(unknown_txt_path, "r") as fh:
       for w in fh.read().splitlines():
         unknown_files.append(unknown_txt_path.parent / w)
    return unknown_files
    

def get_assets(assets):
    apack = AssetPack()
    for asset, cache in assets:
        tf.keras.utils.get_file(origin=asset, untar=True, cache_subdir=cache)
        if Path(cache).name == 'speech_commands':
          apack.background_noise_path = Path(cache) / '_background_noise_'
        elif Path(cache).name == 'unknown_files':
            unknown_files_txt = Path(cache) / "unknown_files.txt"
            apack.unknown_files = pathfix_unknown_files(unknown_files_txt)
    return apack
        
    
        
        
        

def get_examples(keyword, lang='en'):
  ds = None
  for split in SEARCH_SPLITS:
    word_locations = get_word_locations(split)
    if keyword not in word_locations:
      continue


    for location in word_locations[keyword]:
      # download  subsplit, get keyword examples only
      subsplits = get_single_subsplit(location, split)
      new_ds = load_dataset('/content/drive/MyDrive/ml_spoken_words/ml_spoken_words.py', languages=lang, subsplits=subsplits, keyword=keyword)
      
      if ds is None:
        ds = new_ds
      else:
        split = 'validation' if (split == 'dev') else split
        ds[split] = concatenate_datasets([ds[split], new_ds[split]])

  return ds




def get_single_subsplit(i, split):
    subsplits = {
        'train':set(), 
        'dev':set(), 
        'test': set()
    }

    subsplits[split].add(i)
    return subsplits
 

def get_dataset_word_counts(ds: Dataset, split: str):
  if split == 'dev':
    split = 'validation' 

  word_counts = defaultdict(int)
  for ex in ds[split]:
    word_counts[ex['keyword']] += 1
  return word_counts



def get_file_n(root, split) -> int:
  n_files_path = root / split / "n_files.txt"
  with open(n_files_path, 'r') as f:
    n = int(f.read().strip())
  return n


def save_subsplit_counts(lang = 'en'):
  root = Path(f'/content/drive/MyDrive/ml_spoken_words/data/wav/{lang}/')
  for split in ['train', 'dev', 'test']:
    print(f"{lang=}, {split=}")
    n = get_file_n(root, split)

    print(f"getting {n} files from: {root}/{split}")

    for i in range(n):
      subsplits = get_single_subsplit(i, split)
      ds = load_dataset('/content/drive/MyDrive/ml_spoken_words/ml_spoken_words.py', languages='en', subsplits=subsplits)
      word_counts = get_dataset_word_counts(ds, split=split)
      counts_path =  root / split / "word_counts" / f"wordcounts_subsplit_{i}.csv"
      print(f"writing new counts to: {counts_path}")
      with open(counts_path, 'w') as f:
        f.write(f"word,count\n")
        for k, v in word_counts.items():
          f.write(f"{k},{v}\n")



def collate_subsplit_counts(split, lang='en'):
  # create json of {word: subplits}
  subsplit_words = defaultdict(set)
  root = Path(f'/content/drive/MyDrive/ml_spoken_words/data/wav/{lang}/')
  n = get_file_n(root, split)
  for i in range(n):
    path = root / split / "word_counts" / f"wordcounts_subsplit_{i}.csv"
    with open(path, 'r') as f:
      f.readline() # skip header
      for line in f.readlines():
        word, count = line.split(',')
        subsplit_words[word].add(i)

  with open(root / split / 'word_locations.jsonl', 'a') as f:
    for w, n_set in sorted(subsplit_words.items()):
      json.dump({w:list(n_set)}, f)
      f.write('\n')



def get_word_locations(split, lang='en'):
  root = Path(f'/content/drive/MyDrive/ml_spoken_words/data/wav/{lang}/{split}')
  with open(root / 'word_locations.jsonl', 'r') as f:
    json_data = list(f)

  word_locations = defaultdict(list)
  for line in json_data:
    word, n_set = json.loads(line).popitem()
    word_locations[word] = n_set

  return word_locations


def get_samples(ds, few=FEW, split='train'):
  ds = ds.shuffle(seed=42)
  return ds[split][:few]





def listen(filepath):
    IPython.display.display(IPython.display.Audio(filename=filepath, rate="16000"))
