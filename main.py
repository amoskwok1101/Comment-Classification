from dataset import SentimentExample, load_sentiment_examples
from feature import RawTextFeatureExtractor
from model import NetworkModel
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def build_feature(model:str):
    train_exs=load_sentiment_examples('data/train.csv','train',model)
    test_exs=load_sentiment_examples('data/test_v2.csv','test',model)
    feat_extractor=RawTextFeatureExtractor(train_exs,model)
    feat_train=feat_extractor.extract_features(train_exs,'train')
    feat_test=feat_extractor.extract_features(test_exs,'test')
    if model=='LSTM':
        weight=feat_extractor.train_weight()
    else:
        weight=None
    #feat_train in shape(num_of_train_comment,max_len=500), np.array
    #feat_test in shape(num_of_test_comment,max_len=500), np.array
    #weight in shape(num_of_words,embedding_dim=100), torch.Tensor
    
    #use nn.Embedding.from_pretrained(weight,padding_idx=0,freeze=False)
    return train_exs,test_exs,feat_extractor,feat_train,feat_test,weight


def main():
    train_exs,test_exs,feat_extractor,feat_train,feat_test,weight=build_feature('LSTM')
    net=NetworkModel(model='LSTM',weight=weight, num_train_sample = len(train_exs))
    train_labels=[ex.label for ex in train_exs]
    train_labels=np.array(train_labels)
    test_labels=[ex.label for ex in test_exs]
    test_labels=np.array(test_labels)
    net.train(feat_train,train_labels)
    preds=net.predict(feat_test,test_labels)
    
    label_list=['toxic', 'severe_toxic', 'obscene','threat', 'insult', 'identity_hate']
    for i in range(6):
        print('-'*50)
        print((preds[:,i]==1).sum())
        print(f'Accuracy of {label_list[i]}:',accuracy_score(test_labels[:,i],preds[:,i]))
        print(f'Precision of {label_list[i]}:',precision_score(test_labels[:,i],preds[:,i],average='macro'))
        print(f'Recall of {label_list[i]}:',recall_score(test_labels[:,i],preds[:,i],average='macro'))
        print(f'F1 score of {label_list[i]}:',f1_score(test_labels[:,i],preds[:,i],average='macro'))
    

if __name__ == "__main__" :
    main()