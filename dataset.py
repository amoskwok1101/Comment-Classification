import os
import pickle
import re
import warnings
from typing import List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,TweetTokenizer
from tqdm import tqdm
import nltk
#nltk.download("wordnet")

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label=None):
        self.words = words
        self.label = label

    def __repr__(self):
        return f"words: {repr(self.words)}; label: {repr(self.label)}"

    def __str__(self):
        return self.__repr__()
    
    
def load_sentiment_examples(file_path: str, split: str,model: str) -> List[SentimentExample]:
    ''' Read sentiment examples from raw file. Tokenizes and cleans the sentences.'''

    if os.path.exists(f"data/{split}_{model}_exs.pkl"):
        with open(f"data/{split}_{model}_exs.pkl", "rb") as f:
            exs = pickle.load(f)
    else:
        data = pd.read_csv(file_path)
        exs = []

        if split == "train" or split == "valid":
            # For train or valid data
            for row in tqdm(data.itertuples(), total=len(data), desc=f"Load {split} data"):
                label_list=['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']
                label=[getattr(row,label) for label in label_list]
                
                phase = getattr(row, "comment_text")

                # preprocessing
                word_list = text_preprocessing(phase, model)
                if len(word_list) > 0:
                    exs.append(SentimentExample(
                        word_list, label))
        elif split == "test":
            # For test data
            for row in tqdm(data.itertuples(), total=len(data), desc=f"Load {split} data"):
                label_list=['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']
                label=[getattr(row,label) for label in label_list]
                phase = getattr(row, "comment_text")
                # preprocessing
                word_list = text_preprocessing(phase, model)
                if len(word_list) > 0:
                    exs.append(SentimentExample(word_list,label))

        with open(f"data/{split}_{model}_exs.pkl", "wb") as f:
            pickle.dump(exs, f)

    return exs

def text_preprocessing(sentence: str, model: str) -> List[str]:
    '''Preprocess text'''

    # Gets text without tags or markup, remove html
    cur_sentence = BeautifulSoup(sentence, "lxml").get_text()
    # Obtain only letters
    cur_sentence = re.sub("[^a-zA-Z]", " ", cur_sentence)
    # Lower case, tokenization
    cur_sentence = cur_sentence.lower()
    
    
    
    if model=='LSTM':
        words = word_tokenize(cur_sentence.lower())
        preprocessed_words = []
        #lemmatizer = WordNetLemmatizer()
        for word in words:
            # Lemmatizing
            # lemma_word = lemmatizer.lemmatize(word)
            # Remove stop words
            if word not in stopwords.words("english"):
                preprocessed_words.append(word)
    else:
        preprocessed_words = cur_sentence

    return preprocessed_words

def driver():
    model='LSTM'
    train_exs=load_sentiment_examples('data/train.csv','train',model)
    test_exs=load_sentiment_examples('data/test_v2.csv','test',model)
    print(test_exs[0])

if __name__ == "__main__":
    driver()