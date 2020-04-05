#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import collections
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
import pandas as pd
import re
import torch
import torchtext
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from gensim.models import Word2Vec
import time
import copy
from tqdm import tqdm


# In[2]:


# Set Numpy and PyTorch seeds
def set_seeds(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        
# Creating directories
def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# In[3]:


# Arguments
args = Namespace(
    seed=980,
    cuda=True,
    path="training",
    w2vmodel_path="language_n.w2v.model",
    batch_size=64,
    num_workers=4
)
# Set seeds
set_seeds(seed=args.seed, cuda=args.cuda)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))


# In[4]:


path =  Path(args.path)
training_df = pd.read_csv(path / 'train.csv')
test_df = pd.read_csv(path / 'test.csv')
validation_df = pd.read_csv(path / 'valid.csv')

training_df = training_df.filter(regex=("^[a-zA-Z]"))
test_df = test_df.filter(regex=("^[a-zA-Z]"))
validation_df = validation_df.filter(regex=("^[a-zA-Z]"))


# In[5]:


training_df.ix[training_df['flagged']=='N','flagged']=0
training_df.ix[training_df['flagged']=='Y','flagged']=1
test_df.ix[test_df['flagged']=='N','flagged']=0
test_df.ix[test_df['flagged']=='Y','flagged']=1
validation_df.ix[validation_df['flagged']=='N','flagged']=0
validation_df.ix[validation_df['flagged']=='Y','flagged']=1


# In[6]:


from gensim.models import KeyedVectors
w2vmodel = KeyedVectors.load('glove-100d.model')


# In[7]:


stop_words = []
class Vectorizer(object):
    def __init__(self, model):
        self.word_list = model.wv.index2word
        self.vector_list = w2vmodel.wv.vectors
    
    def getVector(self, word):
        try:
            i = self.word_list.index(word)
            return self.vector_list[i]
        except:
            i = self.word_list.index('unknown')
            return self.vector_list[i]
        
    def getVectorsFromText(self, text):
        vectors = []
        for word in text.split(" "):
            if word in stop_words:
                continue
            
            vectors.append(self.getVector(word))
        
        return vectors
    
    # Get id of a word
    def getId(self, word):
        try:
            i = self.word_list.index(word)
            return i
        except:
            i = self.word_list.index('unknown')
            return i
    
    # Tokenize the sentences and replace every token with its id
    def getIdsFromText(self, text):
        ids = []
        for word in text.split(" "):
            if word in stop_words:
                continue
            
            ids.append(self.getId(word))
        
        return ids
    
    def getVectorById(self, id):
        return self.vector_list[id];
    
    # Generate nn.Embedding from trained w2vmodel
    def getEmbedding(self):
        weights = torch.FloatTensor(self.vector_list)
        return torch.nn.Embedding.from_pretrained(weights)
            


# In[8]:


voca = Vectorizer(w2vmodel)


# In[9]:


class ReviewDataset(torch.utils.data.Dataset):
  # rating	reviewUsefulCount	reviewCount	fanCount	mnr	rl	rd	Maximum Content Similarity
    def __init__(self, df, vectorizer: Vectorizer, content_col = 'reviewContent', flagged_col = 'flagged',
                 bf_cols = ['rating','reviewUsefulCount','reviewCount','mnr','rl','rd','Maximum Content Similarity'] ):
        self.df = df
        self.content_col = content_col
        self.bf_cols = bf_cols
        self.flagged_col = flagged_col
        self.vectorizer = vectorizer
    
    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        line = self.df.iloc[idx]
        content = torch.tensor(self.vectorizer.getIdsFromText(line[self.content_col]))
        bf_data = []
        for col in self.bf_cols:
            bf_data.append(line[col])
        bf = torch.tensor(bf_data).reshape(-1)
        flag = torch.tensor(line[self.flagged_col])


        
        return {'content': content, 'bf': bf , 'flag':flag}


# In[10]:


# Create both training and validation datasets

dataframes = {'training': training_df, 'validation': validation_df,'test':test_df}

datasets = {x: ReviewDataset(dataframes[x], voca)
              for x in ['training', 'validation','test']}

dataset_sizes = {x: len(datasets[x]) for x in ['training', 'validation','test']}


# In[11]:


datasets['test'].__getitem__(4)['bf'][1]


# In[12]:


# Costumize `DataLoader` batch format
def variable_size_collate(batch):
    bfs = []
    flags = []
    longest_len = 0
    contents = []
    for item in batch:
        thislen = len(item['content'])
        longest_len = thislen if thislen > longest_len else longest_len
        bfs.append(item['bf'].reshape(-1))
        flags.append(item['flag'])
    
    for i in range(longest_len):
        pos =  []
        for item in batch:
            text = item['content']
            if i < len(text):
                pos.append(text[i])
            else:
                pos.append(torch.tensor(0))
        pos = torch.stack(pos)
        contents.append(pos)

        
    flags = torch.FloatTensor(flags)
    contents = torch.stack(contents)
    bfs = torch.stack(bfs)

    return {'content': contents, 'bf':bfs,'flag': flags}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=variable_size_collate)
              for x in ['training', 'validation','test']}


# In[13]:


embedding = voca.getEmbedding()
embedding


# In[14]:


class LSTM(nn.Module):
    def __init__(self, input_dim,input2_dim,
                 n_layers,embedding_dim,bf_dim,
                 hidden_dim, output_dim, dropout,bidirectional = False):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.fc_bf = nn.Linear(input2_dim, bf_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        
        self.fc = nn.Linear(hidden_dim *  (2 if bidirectional else 1),bf_dim)

        # self.relu = nn.ReLU()

        self.fc_output = nn.Linear(bf_dim * 2, output_dim)
        
    def forward(self, content, bf):

        #text = [sent len, batch size]
        
        embedded = self.embedding(content)
        
        bf_output = self.fc_bf(bf)
        
        embedded_dropout = self.dropout(embedded)
        
        output, hidden = self.rnn(embedded_dropout)

        rnnoutput = self.fc(output[-1,:,:])

        concat = torch.cat((bf_output, rnnoutput),dim=1)

        out = torch.sigmoid(self.fc_output(concat))
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        return out


# In[15]:


INPUT_DIM = embedding.num_embeddings
INPUT2_DIM = 7
EMBEDDING_DIM = embedding.embedding_dim
HIDDEN_DIM = 256
N_LAYERS = 2
BF_DIM = 16
OUTPUT_DIM = 1
DROP_OUT = 0.3
learning_rate = 0.003


# In[16]:


model = LSTM(INPUT_DIM, INPUT2_DIM,N_LAYERS,EMBEDDING_DIM, BF_DIM,HIDDEN_DIM, OUTPUT_DIM, DROP_OUT,bidirectional=True)


# In[17]:


model.to(args.device)
# Replace embedding layer with our trained w2v embedding layer
model.embedding = embedding


# In[18]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))


# In[19]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[20]:


# Here we use MSELoss, in the training process down below, we altered this into RMSE as required
criterion = nn.BCELoss()


# In[21]:


model = model.to(args.device)
criterion = criterion.to(args.device)


# In[22]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# In[23]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            
            contents = batch['content'].to(args.device)
            bfs = batch['bf'].to(args.device)
            flags = batch['flag'].to(args.device)
            
            predictions = model(contents, bfs).squeeze(-1)
            
            loss = criterion(predictions, flags)
            
            acc = binary_accuracy(predictions, flags)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[24]:


def train(model, iterator, optimizer, criterion,scheduler):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        contents = batch['content'].to(args.device)
        bfs = batch['bf'].to(args.device)
        flags = batch['flag'].to(args.device)

        predictions = model(contents, bfs).squeeze(-1)
        
        loss = criterion(predictions, flags)
        
        acc = binary_accuracy(predictions, flags)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    scheduler.step()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[25]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[26]:


N_EPOCHS = 20

losses = []
acces = []

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, dataloaders['training'], optimizer, criterion,lr_scheduler)
    valid_loss, valid_acc = evaluate(model, dataloaders['validation'], criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    torch.save(model.state_dict(), 'model_'+str(epoch)+'.pt')
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-model.pt')

    losses.append(valid_loss)
    
    acces.append(valid_acc*100)
    
    print('Epoch: %d \t Epoch Time: %d m %d s' % (epoch+1,epoch_mins,epoch_secs))
    print('\tTrain Loss: %.3f \t  Train Acc: %.2f%%' %(train_loss,train_acc))
    print('\t Val. Loss: %.3f \t  Val. Acc: %.2f%%' %(valid_loss,valid_acc))


# In[27]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
sns.set(rc={'figure.figsize':(15, 10)})


# In[28]:


plt.plot(losses)
plt.savefig("valid_losses.jpg")


# In[29]:


plt.plot(acces)
plt.savefig("valid_acces.jpg")


# In[30]:


model.load_state_dict(torch.load('best-model.pt'))

test_loss, test_acc = evaluate(model, dataloaders['test'], criterion)

print('Test. Loss: %.3f \t  Test. Acc: %.2f%%' %(test_loss,test_acc))


# In[ ]:




