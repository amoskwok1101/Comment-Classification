import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertPreTrainedModel,logging, DistilBertModel
logging.set_verbosity_warning()
class BertClassifier(nn.Module):
    
    def __init__(self, bert: DistilBertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, inputs_embeds=None, position_ids=None, head_mask=None, labels=None):
        
        attention_mask=(input_ids!=0).float()
        outputs = self.bert(input_ids,attention_mask=attention_mask)
        
        cls_output = outputs[0][:,0] # batch, hidden
        cls_output = self.classifier(cls_output) # batch, 6
        cls_output = torch.sigmoid(cls_output)
        return cls_output