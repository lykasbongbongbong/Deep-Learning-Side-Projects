import os
import torch
import torch.utils.data as data

class DataTransformer:
    def __init__(self):
        self.char2idx=self.build_char2idx()  # {'SOS':0,'EOS':1,'a':2,'b':3 ... 'z':27}
        self.idx2char=self.build_idx2char()  # {0:'SOS',1:'EOS',2:'a',3:'b' ... 27:'z'}
        self.tense2idx={'sp':0,'tp':1,'pg':2,'p':3}
        self.idx2tense={0:'sp',1:'tp',2:'pg',3:'p'}
        self.max_length=0  # max length of the training data word(contain 'EOS')

    def build_char2idx(self):
        state_dict={'SOS':0,'EOS':1}
        state_dict.update([(chr(i+97),i+2) for i in range(0,26)])
        return state_dict

    def build_idx2char(self):
        state_dict={0:'SOS',1:'EOS'}
        state_dict.update([(i+2,chr(i+97)) for i in range(0,26)])
        return state_dict

    def string2tensor(self,string,add_eos=True):
        indices=[self.char2idx[char] for char in string]
        if add_eos:
            indices.append(self.char2idx['EOS'])
        return torch.tensor(indices,dtype=torch.long).view(-1,1)

    def tense2tensor(self,tense):
        return torch.tensor([tense],dtype=torch.long)

    def tensor2string(self,tensor):
        string = str()
        string_length=tensor.size(0)
        for i in range(string_length):
            char_state = self.idx2char[tensor[i].item()]
            if char_state=='EOS':
                break
            string+=char_state
        return string

    def get_dataset(self,path,train):
        words = list()
        tenses = list()
        with open(path,'r') as file:
            if train:
                for line in file:
                    words.extend(line.split('\n')[0].split(' '))
                    tenses.extend(range(0,4))
            else:
                for line in file:
                    words.append(line.split('\n')[0].split(' '))
                test_tenses=[['sp','p'],['sp','pg'],['sp','tp'],['sp','tp'],['p','tp'],['sp','pg'],['p','sp'],['pg','sp'],['pg','p'],['pg','tp']]
                for test_tense in test_tenses:
                    tenses.append([self.tense2idx[tense] for tense in test_tense])
        return words,tenses

class WordDataset(data.Dataset):
    def __init__(self,path,train):
        self.train = train
        self.dataTransformer=DataTransformer()
        self.words,self.tenses=self.dataTransformer.get_dataset(os.path.join('dataset',path),train)
        self.max_length=self.get_max_length(self.words)
        self.string2tensor = self.dataTransformer.string2tensor  
        self.tense2tensor = self.dataTransformer.tense2tensor   
        self.tensor2string = self.dataTransformer.tensor2string 

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        if self.train:
            return self.string2tensor(self.words[idx],add_eos=True),self.tense2tensor(self.tenses[idx])
        else:
            return self.string2tensor(self.words[idx][0],add_eos=True),self.tense2tensor(self.tenses[idx][0]),\
                   self.string2tensor(self.words[idx][1],add_eos=True),self.tense2tensor(self.tenses[idx][1])

    def get_max_length(self,words):
        max_length=0
        for word in words:
            max_length=max(max_length,len(word))
        return max_length