import torch
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np

from lstm import LSTM
from transformers import BertModel,BertPreTrainedModel,logging,DistilBertModel,AdamW, get_linear_schedule_with_warmup
from bert import  BertClassifier
from tqdm import tqdm
import itertools
import warnings
import os
import pickle
warnings.filterwarnings("ignore")
logging.set_verbosity_warning()

class NetworkModel:
    def __init__(self, lr=1e-3, batch_size=4, test_batch_size=4, num_epoch=5,
                 num_classes=6, model='LSTM',num_train_sample = 159502, weight=None,ratio=0.5,sample=True):
        self.lr = lr
        self.num_train_sample = num_train_sample 
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_epoch = num_epoch
        self.modelType = model
        self.model = None
        self.ratio=ratio
        self.loss = nn.BCELoss()
        self.device = 'cuda'  # 'cuda' if torch.cuda.is_available() else
        self.sample=sample
        if self.modelType == 'LSTM':
            self.model = LSTM(weight.to(self.device)).to(self.device)
            self.optim=optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.modelType == "BERT":
            num_training_steps = (num_train_sample // self.batch_size) * num_epoch
            #self.model = BertClassifier(BertModel.from_pretrained('bert-base-cased'),6)
            self.model = BertClassifier(DistilBertModel.from_pretrained('distilbert-base-uncased'),1).to(self.device)
            self.optim=AdamW(self.model.parameters(), lr=2e-5)
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=0,num_training_steps=num_training_steps)

    def train(self, train_input_ids,train_targets) -> None:
        num_samples=int(len(train_input_ids)*self.ratio)
        if os.path.exists(f"data/weight_sample_{self.modelType}.pkl"):
            with open(f"data/weight_sample_{self.modelType}.pkl", "rb") as f:
                weight = pickle.load(f)
        else:
            
            classes=list(itertools.product([1,0], repeat=6))
            classes=[list(i) for i in classes]
            classes=np.array(classes)
            class_freq=[]
            class_=np.zeros(len(train_input_ids))
            for i in range(len(classes)):
                sum_=0
                for j in range(len(train_targets)):
                    if np.all(train_targets[j]==classes[i]):
                        sum_+=1
                        class_[j]=i
                class_freq.append(sum_)
            weight=1/np.array(class_freq)
            weight=np.where(weight!=np.inf,weight,0)
            weight=weight[class_.astype(int)]
            weight=weight/np.sum(weight)
            with open(f"data/weight_sample_{self.modelType}.pkl", "wb") as f:
                pickle.dump(weight, f)
                
        for epoch in range(self.num_epoch):
            running_correct=0
            running_loss = 0.0
            running_nsample = 0.0
            if self.sample==True:
                index=np.random.choice(len(train_input_ids),num_samples,p=weight,replace=True)
            else:
                index=np.random.choice(len(train_input_ids),num_samples,replace=False)
            train_input_ids_sample = torch.from_numpy(train_input_ids[index]).long().to(self.device)
            train_targets_sample = torch.from_numpy(train_targets[index]).float().to(self.device)
            since=time.time()
            
            self.model.train()
            for step in tqdm(range(num_samples // self.batch_size + 1), desc=f"Epoch {epoch}", leave=False):
                # shuffle current data
                cur_idxs = torch.randperm(num_samples)
                train_input_ids_sample = train_input_ids_sample[cur_idxs]
                train_targets_sample = train_targets_sample[cur_idxs]

                left = step * self.batch_size
                right = min((step+1) * self.batch_size, num_samples)
                batch_ids = train_input_ids_sample[left:right]
                batch_y = train_targets_sample[left:right]

                
                output = self.model(batch_ids.to(self.device))
                
                loss = self.loss(output,batch_y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                running_loss += loss.item() * (right - left)
                running_nsample += (right - left)
                running_correct+= (torch.round(output)==batch_y).sum().item()
                

            epoch_loss = running_loss/running_nsample
            epoch_acc = running_correct/(running_nsample*6)
            
            print(f"Epoch: {epoch + 1}/{self.num_epoch}, Loss: {epoch_loss:.5f}, "
                  f"Acc: {epoch_acc * 100:.2f}, Time: {time.time() - since:.1f}s")
            print("-" * 40)
        torch.save(self.model.state_dict(), f'{self.modelType}.pt')

    def predict(self, test_feat, test_label):
        test_feat = torch.from_numpy(test_feat).long()
        test_label = torch.from_numpy(test_label).float()

        self.model.eval()

        test_loss = 0.
        preds = []
        num_samples = len(test_feat)
        with torch.no_grad():
            for step in tqdm(range(len(test_feat)// self.batch_size + 1), desc=f"Evaluate"):
                

                    left = step * self.test_batch_size
                    right = min((step+1) * self.test_batch_size, num_samples)
                    batch_ids = test_feat[left:right].to(self.device)
                    batch_y = test_label[left:right].to(self.device)
                    
                    output=self.model(batch_ids)
                    loss=self.loss(output,batch_y)
                    batch_pred=torch.round(output)
                    test_loss += loss.item() * (right - left)
                    #running_correct+= (batch_pred==batch_y).sum().item()
                    
                    for pred in batch_pred:
                        preds.append(pred.cpu().data.numpy())
                
        test_loss /= len(test_feat)
        preds=np.array(preds)
        print(f"Test loss: {test_loss:.5f}")
        return preds