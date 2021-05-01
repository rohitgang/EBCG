import torch
class TextGen(torch.nn.ModuleList):

    def __init__(self, args, vocab_size):
        super(TextGen, self).__init__()
        # batch = args['batch_size']
        self.batch_size = args['batch_size']
        self.hidden_dim = args['hidden_dim']
        self.input_size = vocab_size
        self.num_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = args['window']
        self.init_method = args['init_method']

        self.dropout = torch.nn.Dropout(0.3)
        
        self.embedding = torch.nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        
        self.lstm_cell_forward = torch.nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_cell_backward = torch.nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        
        self.lstm_cell = torch.nn.LSTMCell(self.hidden_dim*2, self.hidden_dim*2)
        
        self.linear = torch.nn.Linear(self.hidden_dim*2, self.num_classes)
        
    def forward(self, input_) :
        """
            Hidden states
        """
        hidden_state_forward = torch.zeros(input_.size(0), self.hidden_dim)
        hidden_state_backward = torch.zeros(input_.size(0), self.hidden_dim)
        
        """
            Cell states
        """
        cell_state_forward = torch.zeros(input_.size(0), self.hidden_dim)
        cell_state_backward = torch.zeros(input_.size(0), self.hidden_dim)
        
        """
            LSTM states
        """
        hidden_state_LSTM = torch.zeros(input_.size(0), self.hidden_dim*2)
        cell_state_LSTM = torch.zeros(input_.size(0), self.hidden_dim*2)
        
        """
            Init weights
        """
        
        if(self.init_method == 'xavier') :
            torch.nn.init.xavier_normal_(hidden_state_forward)
            torch.nn.init.xavier_normal_(hidden_state_backward)
            torch.nn.init.xavier_normal_(hidden_state_LSTM)
            torch.nn.init.xavier_normal_(cell_state_forward)
            torch.nn.init.xavier_normal_(cell_state_backward)
            torch.nn.init.xavier_normal_(cell_state_LSTM)
            
        if(self.init_method == 'kaiming') :
            torch.nn.init.kaiming_normal_(hidden_state_forward)
            torch.nn.init.kaiming_normal_(hidden_state_backward)
            torch.nn.init.kaiming_normal_(hidden_state_LSTM)
            torch.nn.init.kaiming_normal_(cell_state_forward)
            torch.nn.init.kaiming_normal_(cell_state_backward)
            torch.nn.init.kaiming_normal_(cell_state_LSTM)
            
        output = self.embedding(input_)
        output = output.view(self.sequence_len, input_.size(0), -1)
        
        forward, backward = [], []
        
        for i in range(self.sequence_len):
            hidden_state_forward, cell_state_forward = self.lstm_cell_forward(output[i],
                                                                              (hidden_state_forward,
                                                                               cell_state_forward))
            hidden_state_forward = self.dropout(hidden_state_forward)
            cell_state_forward = self.dropout(cell_state_forward)
            
            forward.append(hidden_state_forward)
        
        for i in reversed(range(self.sequence_len)):
            hidden_state_backward, cell_state_backward = self.lstm_cell_backward(output[i],
                                                                                 (hidden_state_backward,
                                                                                  cell_state_backward))
            hidden_state_backward = self.dropout(hidden_state_backward)
            cell_state_backward = self.dropout(cell_state_backward)
            
            backward.append(hidden_state_backward)
            
        for f, b in zip(forward, backward):
            input_tensor = torch.cat((f,b), 1)
            hidden_state_LSTM, cell_state_LSTM = self.lstm_cell(input_tensor, (hidden_state_LSTM,
                                                                               cell_state_LSTM))
            
        output = self.linear(hidden_state_LSTM)
        
        return output