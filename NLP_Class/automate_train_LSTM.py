from process_data import Data_Processor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import TextGen
import torch 
from tqdm.notebook import tqdm

data = pd.read_pickle('event_data.pickle')
data_proc = Data_Processor(data['commentary'], 100)
seq = data_proc.seq
targets = data_proc.targets
seq_tr, seq_te, target_tr, target_te = train_test_split(seq, targets, 
                                                        test_size=0.33, random_state=42)

args = {'batch_size' : 128,
        'hidden_dim' : 128,
        'window' : 100,
        'init_method' : 'xavier'}

num_epochs = 20
lr = 0.001

gen = TextGen(args, data_proc.vocab_size)

optimizer = torch.optim.RMSprop(gen.parameters(), lr=0.001)

batches = int(len(seq_tr) / args['batch_size'])

gen.train()

for epoch in tqdm(range(num_epochs)):
    
    for i in range(batches):
        
        try :
            x_batch = seq_tr[i * args['batch_size'] : (i+1) * args['batch_size']]
            y_batch = target_tr[i * args['batch_size'] : (i+1) * args['batch_size']]
        except:
            x_batch = seq_tr[i * args['batch_size'] :]
            y_batch = target_tr[i * args['batch_size'] :]

        optimizer.zero_grad()

        x = torch.from_numpy(x_batch).type(torch.LongTensor)
        y = torch.from_numpy(y_batch).type(torch.LongTensor)
        y_pred = gen(x)
        
        loss = torch.nn.functional.cross_entropy(y_pred, y.squeeze())

        loss.backward()
        
        optimizer.step()

torch.save(gen.state_dict(), 'weights/textgen_model3.pt')