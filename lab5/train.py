from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from util import loss_function, compute_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token=0
EOS_token=1



def train(model,train_loader,optimizer,teacher_forcing_ratio,kl_weight,tensor2string):
    model.train()
    total_CEloss=0
    total_KLloss=0
    total_BLEUscore=0
    loss = 0
    optimizer.zero_grad()

    for word_tensor,tense_tensor in train_loader:
        optimizer.zero_grad()
        word_tensor,tense_tensor=word_tensor[0],tense_tensor[0]
        word_tensor,tense_tensor=word_tensor.to(device),tense_tensor.to(device)

        h0 = model.encoder.init_h0(model.hidden_size - model.conditional_size)
        c = model.tense_embedding(tense_tensor).view(1, 1, -1)
        encoder_hidden_state = torch.cat((h0, c), dim=-1)
        encoder_cell_state = model.encoder.init_c0()

        if random.random() < teacher_forcing_ratio:
            # Teacher forcing: Feed the target as the next input
            use_teacher_forcing = True 
        else:
            # Without teacher forcing: use its own predictions as the next input
            use_teacher_forcing = False

        output, distribution, mean, logvar = model(word_tensor,encoder_hidden_state,encoder_cell_state,c,use_teacher_forcing)
        CEloss,KLloss = loss_function(distribution,output.size(0),word_tensor.view(-1),mean,logvar)
        loss = CEloss + kl_weight * KLloss
        total_CEloss += CEloss.item()
        total_KLloss += KLloss.item()
        predict,target = tensor2string(output),tensor2string(word_tensor)
        total_BLEUscore += compute_bleu(predict,target)

        loss.backward()
        optimizer.step()

        avg_CEloss = total_CEloss/len(train_loader)
        avg_KLloss = total_KLloss/len(train_loader)
        avg_BLEUscore = total_BLEUscore/len(train_loader)

    return avg_CEloss, avg_KLloss, avg_BLEUscore



def evaluate(model,test_loader,tensor2string):
    """
    :param tensor2string: function(tensor){ return string }  (cutoff EOS automatically)
    :return: [[input,target,predict],[input,target,predict]...], BLEUscore
    """
    model.eval()
    output_list = list()
    total_BLEUscore=0
    with torch.no_grad():
        for in_word_tensor,in_tense_tensor,tar_word_tensor,tar_tense_tensor in test_loader:
            in_word_tensor,in_tense_tensor=in_word_tensor[0].to(device),in_tense_tensor[0].to(device)
            tar_word_tensor,tar_tense_tensor=tar_word_tensor[0].to(device),tar_tense_tensor[0].to(device)

            h0 = model.encoder.init_h0(model.hidden_size - model.conditional_size)
            in_c = model.tense_embedding(in_tense_tensor).view(1, 1, -1)
            encoder_hidden_state = torch.cat((h0, in_c), dim=-1)
            encoder_cell_state = model.encoder.init_c0()

            tar_c=model.tense_embedding(tar_tense_tensor).view(1,1,-1)
            output = model.inference(in_word_tensor,encoder_hidden_state,encoder_cell_state,tar_c)
            target_word=tensor2string(tar_word_tensor)
            predict_word=tensor2string(output)
            output_list.append([tensor2string(in_word_tensor),target_word,predict_word])
            total_BLEUscore+=compute_bleu(predict_word,target_word)

    avg_BLEUscore = total_BLEUscore / len(test_loader)
    return output_list, avg_BLEUscore

def Word_generation(model,latent_size,tensor2string):
    model.eval()
    output_list = list()
    with torch.no_grad():
        for i in range(100):
            latent = torch.randn(1, 1, latent_size).to(device)
            tmp = []
            for tense in range(4):
                word = tensor2string(model.generate(latent, tense))
                tmp.append(word)
            output_list.append(tmp)

    return output_list