from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token=0
EOS_token=1

class VAE(nn.Module):  # conditional VAE
    class Encoder(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(VAE.Encoder,self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.rnn = nn.LSTM(hidden_size, hidden_size)

        def forward(self, input, hidden_state, cell_state):
            embedded = self.embedding(input).view(1,1,-1) 
            output, (hidden_state, cell_state) = self.rnn(embedded, (hidden_state, cell_state))
            return output, hidden_state, cell_state

        def init_h0(self,size):
            return torch.zeros(1, 1, size, device=device)

        def init_c0(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)

    class Decoder(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(VAE.Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.rnn = nn.LSTM(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, input_size)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, input, hidden_state, cell_state):
            output = self.embedding(input).view(1, 1, -1)
            output = F.relu(output)
            output, (hidden_state, cell_state) = self.rnn(output, (hidden_state, cell_state))
            output = self.softmax(self.out(output[0]))
            return output, hidden_state, cell_state

        def init_h0(self):
            pass

        def init_c0(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)

    def __init__(self, input_size, hidden_size, latent_size, conditional_size,max_length):
        super(VAE,self).__init__()
        self.encoder = self.Encoder(input_size, hidden_size)
        self.decoder = self.Decoder(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.conditional_size = conditional_size
        self.max_length=max_length
        self.tense_embedding = nn.Embedding(4, conditional_size) 
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        self.latentcondition2hidden=nn.Linear(latent_size+conditional_size,hidden_size)

    def forward(self,input_tensor,encoder_hidden_state,encoder_cell_state,c,use_teacher_forcing):
        input_length = input_tensor.size(0)
        for ei in range(input_length):
            _ ,encoder_hidden_state,encoder_cell_state=self.encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
       
        mean=self.hidden2mean(encoder_hidden_state)
        logvar=self.hidden2logvar(encoder_hidden_state)

        #do sampling
        latent=self.reparameterize(mean,logvar)
        decoder_hidden_state = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell_state = self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)


        predict_distribution=torch.zeros(input_length,self.input_size,device=device)
        predict_output = None
        for di in range(input_length):
            output,decoder_hidden_state,decoder_cell_state=self.decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            predict_distribution[di]=output[0]
            predict_class=output.topk(1)[1]
            predict_output=torch.cat((predict_output,predict_class)) if predict_output is not None else predict_class

            if use_teacher_forcing:  
                decoder_input=input_tensor[di]
            else:
                if predict_class.item() == EOS_token:
                    break
                decoder_input = predict_class

        return predict_output,predict_distribution,mean,logvar

    def inference(self,input_tensor,encoder_hidden_state,encoder_cell_state,c):

        input_length=input_tensor.size(0)
        for ei in range(input_length):
            _ ,encoder_hidden_state,encoder_cell_state=self.encoder(input_tensor[ei],encoder_hidden_state,encoder_cell_state)
      
        mean=self.hidden2mean(encoder_hidden_state)
        logvar=self.hidden2logvar(encoder_hidden_state)
        # sampling a point
        latent=self.reparameterize(mean,logvar)
        decoder_hidden_state = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell_state = self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)

        predict_output = None
        for di in range(self.max_length):
            output, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input,decoder_hidden_state,decoder_cell_state)
            predict_class = output.topk(1)[1]
            predict_output = torch.cat((predict_output, predict_class)) if predict_output is not None else predict_class

            if predict_class.item() == EOS_token:
                break
            decoder_input = predict_class

        return predict_output

    def generate(self,latent,tense):
        tense_tensor=torch.tensor([tense]).to(device)
        c=self.tense_embedding(tense_tensor).view(1, 1, -1)
        decoder_hidden_state = self.latentcondition2hidden(torch.cat((latent, c), dim=-1))
        decoder_cell_state=self.decoder.init_c0()
        decoder_input = torch.tensor([[SOS_token]], device=device)

        predict_output = None
        for di in range(self.max_length):
            output, decoder_hidden_state, decoder_cell_state = self.decoder(decoder_input, decoder_hidden_state, decoder_cell_state)
            predict_class = output.topk(1)[1]
            predict_output = torch.cat((predict_output, predict_class)) if predict_output is not None else predict_class

            if predict_class.item() == EOS_token:
                break
            decoder_input = predict_class

        return predict_output

    def reparameterize(self,mean,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        latent=mean+eps*std
        return latent



