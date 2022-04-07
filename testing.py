import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
from sklearn import utils
from numpy.linalg import norm
import collections

model = torch.load('model_50.pth', map_location=torch.device('cpu'))

x_test = np.load('ret.npy')

pred = model(torch.tensor(x_test).view(-1,1,48000).float())

#cosine similarity
def cosine_similarity(embed1, embed2):
    embed1 = embed1.detach().numpy()
    embed2 = embed2.detach().numpy()
    norm1 = norm(embed1, axis=-1)
    norm2 = norm(embed2, axis=-1)
    de_norm = norm1 * norm2

    no_mult = np.sum(np.multiply(embed1, embed2), axis=-1)
    s = np.true_divide(no_mult, de_norm)

    return s
  
#check embedds for one person
c = collections.Counter()
for emb in pred:
    c[round(cosine_similarity(pred[0], emb),1)]+=1
    
x__=[]
y__=[]
for item in c.items():
    x__.append(item[0])
    y__.append(item[1])
    
plt.bar(x__, y__)
plt.show()
