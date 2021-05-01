"""
    Prepare the data
"""
import string
import numpy as np
class Data_Processor:


    def __init__(self, text, window_size, other_symbols=None):
        
        self.raw_text = text
        self.WINDOW = window_size
        self.letters = list(string.ascii_lowercase)
        if other_symbols : 
            for symbol in other_symbols:
                self.letters.append(symbol)
        else : self.letters.append(' ')
        self.text = self.break_down()
        self.ch2idx = {}
        self.idx2ch = {}
        self.create_dict()
        self.seq, self.targets = self.create_seq()
        self.vocab_size = len(self.ch2idx)
        
    def break_down(self):

        str_ = ''

        text = [wrd.lower() for line in self.raw_text for wrd in line.split('.')]
        for word in text:
            str_ += word.strip()

        arr_ = list()
        for char in str_:
            arr_.append(char)

        str_ = [ch for ch in arr_ if ch in self.letters]

        return str_
    
    def create_dict(self):
        
        idx = 0

        for ch in self.text:
            if ch not in self.ch2idx.keys():

                self.ch2idx[ch] = idx
                self.idx2ch[idx] = ch
                idx += 1
    
    def create_seq(self):

        x, y = list(), list()

        for i in range(len(self.text)):

            try:
                seq = self.text[i:i+self.WINDOW]
                seq = [self.ch2idx[ch] for ch in seq]

                target = self.text[i+self.WINDOW]
                target = self.ch2idx[target]

                x.append(seq)
                y.append(target)

            except: pass

        x = np.array(x)
        y = np.array(y)

        return x, y