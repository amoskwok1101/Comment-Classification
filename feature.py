from abc import abstractmethod
from typing import List
import os
import pickle
import numpy as np
from tqdm import tqdm
from dataset import SentimentExample
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from transformers import AutoTokenizer


class FeatureExtractor(object):
    def __init__(self):
        raise NotImplementedError("Don't call me, call my subclasses")

    @abstractmethod
    def extract_features(self, exs: List[SentimentExample]) -> np.array:
        raise NotImplementedError("Don't call me, call my subclasses")


class RawTextFeatureExtractor(FeatureExtractor):
    def __init__(self, train_exs: List[SentimentExample],model) -> None:
        # Construct a vocabulary
        self.max_len=1000
        self.max_vocab=1000000
        self.model=model
        self.tokenizer=None
        
        if self.model=='LSTM':
            if os.path.exists(f"data/vocabulary.pkl"):
                self.max_len = 0
                for ex in tqdm(train_exs,total=len(train_exs), desc=f"Building Vocabulary"):
                    self.max_len = max(self.max_len, len(ex.words))
                with open(f"data/vocabulary.pkl", "rb") as f:
                    self.vocab = pickle.load(f)
            else:

                all_words = dict()
                #self.max_len = 0
                for ex in tqdm(train_exs,total=len(train_exs), desc=f"Building Vocabulary"):
                    for i in ex.words:
                        if i in all_words:
                            all_words[i]+=1
                        else:
                            all_words[i]=1
                sorted_words=dict(sorted(all_words.items(), key=lambda x:x[1] ))
                del all_words
                self.vocab=dict()
                i=1
                for word in sorted_words.keys():
                    if i>self.max_vocab:
                        break
                    self.vocab[word] = i
                    i+=1
                    
                self.vocab["<unk>"] = self.max_vocab+1
                self.vocab["<pad>"] = 0
                with open(f"data/vocabulary.pkl", "wb") as f:
                    pickle.dump(self.vocab, f)
        else:
            self.tokenizer=AutoTokenizer.from_pretrained("distilbert-base-cased")

    def extract_features(self, exs: List[SentimentExample], type_: str) -> np.array:
        feats = []
        if os.path.exists(f"data/{type_}_{self.model}_feat.pkl"):
            with open(f"data/{type_}_{self.model}_feat.pkl", "rb") as f:
                feats = pickle.load(f)
        else:
            if self.model=='LSTM':
                for ex in tqdm(exs,total=len(exs), desc=f"Extract features"):
                    feat = []
                    for word in ex.words:
                        if word in self.vocab:
                            feat.append(self.vocab[word])
                        else:
                            feat.append(self.vocab["<unk>"])
                    feats.append(feat)
                
                feats=pad_sequences(feats, maxlen = self.max_len,padding='post')
            else:
                sentences=[ex.words for ex in exs]
                #tokens=self.tokenizer(sentences, padding=True, truncation=True,max_length=100)
                tokens=self.tokenizer(sentences, padding=True, truncation=True)
                feats=np.array(tokens.input_ids)
                    
                
            with open(f"data/{type_}_{self.model}_feat.pkl", "wb") as f:
                pickle.dump(feats, f)

        return feats
    def train_weight(self):
        # load pretrained word2vec 100d embeddings
        
        if os.path.exists(f"data/weight.pkl"):
            weight=torch.Tensor()
            with open(f"data/weight.pkl", "rb") as f:
                weight = pickle.load(f)
        else:
            weight=np.zeros((len(self.vocab),100))
            with open(f"data/wv.pkl", "rb") as f:
                wv_from_bin = pickle.load(f)
            
            
            for i in tqdm(self.vocab.keys(),total=len(self.vocab),desc="Building Weight"):
                try:
                    weight[self.vocab[i]]=wv_from_bin.get_vector(i)
                except:
                    pass
            weight=torch.Tensor(weight)    
            with open(f"data/weight.pkl", "wb") as f:
                pickle.dump(weight, f)
        return weight
                
    
def driver():
    from dataset import SentimentExample, load_sentiment_examples
    model='LSTM'
    train_exs=load_sentiment_examples('data/train.csv','train',model)
    test_exs=load_sentiment_examples('data/test_v2.csv','test',model)
    feat_extractor=RawTextFeatureExtractor(train_exs,model)
    feat_train=feat_extractor.extract_features(train_exs,'train')
    feat_test=feat_extractor.extract_features(test_exs,'test')
    weight=feat_extractor.train_weight()
    print(feat_train.shape)

if __name__ == "__main__":
    driver()
    