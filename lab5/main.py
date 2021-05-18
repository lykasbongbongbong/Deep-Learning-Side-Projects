from __future__ import unicode_literals, print_function, division
import os
import torch
from torch import optim
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import WordDataset
from model import VAE
from train import train, evaluate, Word_generation
from util import get_teacher_forcing_ratio,get_kl_weight,get_gaussian_score,plot
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
input_size = 28   
hidden_size = 256 
latent_size = 32
conditional_size = 8
LR = 0.05
epochs = 200
kl_annealing_type='monotonic' #monotonic/cyclical
time = 2
batch_size = 1
model_save_path = os.path.join("weights", kl_annealing_type+".weights")
fig_save_path = os.path.join("result", kl_annealing_type+".png")


def main():
    # dataloader
    trainset = WordDataset(path='train.txt',train=True)
    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    testset = WordDataset(path='test.txt',train=False)
    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=False)
    # VAE model
    vae=VAE(input_size,hidden_size,latent_size,conditional_size,max_length=trainset.max_length).to(device)
    # train
    optimizer = optim.SGD(vae.parameters(), lr=LR)
    Cross_Entropy_loss_all = list()
    KL_Loss_all = list()
    BLEU_Score_all = list()
    teacher_forcing_ratio_all = list()
    KL_weight_all = list() 
    best_BLEU = 0
    best_model = None
    for epoch in tqdm(range(1,epochs+1)):
        # train
        teacher_forcing_ratio = get_teacher_forcing_ratio(epoch,epochs)
        kl_weight = get_kl_weight(epoch,epochs,kl_annealing_type,time)
        CEloss,KLloss,_=train(vae, train_loader, optimizer, teacher_forcing_ratio, kl_weight, trainset.tensor2string)
        Cross_Entropy_loss_all.append(CEloss)
        KL_Loss_all.append(KLloss)
        teacher_forcing_ratio_all.append(teacher_forcing_ratio)
        KL_weight_all.append(kl_weight)
        print(f'epoch{epoch}  tf_ratio:{teacher_forcing_ratio:.4f}  kl_weight:{kl_weight:.4f}')
        print(f'Cross Entropy Loss:{CEloss:.4f} + KL:{KLloss:.4f} = {CEloss+KLloss:.4f}')

        # test
        conversion,BLEUscore=evaluate(vae,test_loader,testset.tensor2string)
        BLEU_Score_all.append(BLEUscore)
        print(conversion)
        print(f'BLEU score:{BLEUscore:.4f}') 
      
        if BLEUscore> best_BLEU:
            best_BLEU = BLEUscore
            best_model = vae.state_dict()
            torch.save(best_model , model_save_path)
           
    torch.save(best_model,os.path.join(model_save_path))
    fig = plot(epochs,Cross_Entropy_loss_all,KL_Loss_all,BLEU_Score_all,teacher_forcing_ratio_all,KL_weight_all)
    fig.savefig(os.path.join(fig_save_path))

if __name__=='__main__':
    main()
    

    

    