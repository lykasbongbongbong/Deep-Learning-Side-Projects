import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device=device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_function(predict_distribution,predict_output_length,target,mu,logvar):
    Loss = nn.CrossEntropyLoss()
    CEloss = Loss(predict_distribution[:predict_output_length],target[:predict_output_length])
    KLloss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return CEloss, KLloss

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)
    
def get_teacher_forcing_ratio(epoch,epochs):
    # from 1.0 to 0.0
    teacher_forcing_ratio = 1.-(1./(epochs-1))*(epoch-1)
    return teacher_forcing_ratio

def get_kl_weight(epoch,epochs,kl_annealing_type,time):
    """
    :param epoch: i-th epoch
    :param kl_annealing_type: 'monotonic' or 'cycle'
    :param time:
        if('monotonic'): # of epoch for kl_weight from 0.0 to reach 1.0
        if('cycle'):     # of cycle
    """
    assert kl_annealing_type=='monotonic' or kl_annealing_type=='cycle','kl_annealing_type not exist!'

    if kl_annealing_type == 'monotonic':
        return (1./(time-1))*(epoch-1) if epoch<time else 1.

    else: #cycle
        period = epochs//time
        epoch %= period
        KL_weight = sigmoid((epoch - period // 2) / (period // 10)) / 2
        return KL_weight

def get_gaussian_score(words):
    words_list = []
    score = 0
    yourpath = "dataset/train.txt"  #should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def plot(epochs,CEloss_list,KLloss_list,BLEUscore_list,teacher_forcing_ratio_list,kl_weight_list):
    x=range(1,epochs+1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x,CEloss_list, label='Cross Entropy loss')
    plt.plot(x,KLloss_list, label='KL loss')
    plt.plot(x,BLEUscore_list,label='BLEU score')
    plt.plot(x,teacher_forcing_ratio_list,linestyle=':',label='teacher force ratio')
    plt.plot(x,kl_weight_list,linestyle=':',label='KL weight')
    plt.legend()

    return fig


