import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import MultiheadAttention
import math

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=100, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class CNN_RNNEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dropout = cfgs.drop_rate
        
        self.trigram_conv = nn.Conv1d(cfgs.hidden_dim, cfgs.hidden_dim, 3, stride=1, padding=2, dilation=2)
        self.bilstm = nn.LSTM(input_size=cfgs.hidden_dim,
                              hidden_size=cfgs.hidden_dim//2,
                              num_layers=2,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)

    def forward(self, x, mask):
        length = mask.sum(dim=-1)
         
        conv_input = x.permute(0, 2, 1) # 1, 128, 63
        conv_output = self.trigram_conv(conv_input)
        conv_output = conv_output.permute(0, 2, 1) # 1, 63,128
        feat, _ = self.bilstm(conv_output)
        return feat

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):

    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

class TransformerEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

        self.self_attn = MultiheadAttention(cfgs.hidden_dim, cfgs.attention_heads)
        self.feed_forward = PositionwiseFeedForward(cfgs.hidden_dim)
        self.layer_norm = nn.LayerNorm(cfgs.hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(cfgs.drop_rate)
        self.transformer_layers = cfgs.transformer_layers
    
    def forward(self, inputs , mask):
        out = inputs
        for i in range(self.transformer_layers):
            input_norm = self.layer_norm(out)
            #mask = mask.unsqueeze(1)
            context,_ = self.self_attn(input_norm, input_norm, input_norm,
                                 key_padding_mask=mask)
            out = self.dropout(context) + inputs
            out = self.feed_forward(out)
        return out

class LSTMContex(nn.Module):

    def __init__(self, cfgs):
        super(LSTMContex, self).__init__()
        self.cfgs = cfgs

        self.lstm = nn.LSTM(self.cfgs.hidden_dim, self.cfgs.hidden_dim)

    def forward(self, feat_input, mask):
        rep, _ = self.lstm(feat_input)
        return rep

class CNNEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dropout = cfgs.drop_rate      
        self.trigram_conv = nn.Conv1d(cfgs.hidden_dim, cfgs.hidden_dim, 3, stride=1, padding=2, dilation=2)

    def forward(self, x, mask):
        length = mask.sum(dim=-1)
         
        conv_input = x.permute(0, 2, 1) # 1, 128, 63
        conv_output = self.trigram_conv(conv_input)
        conv_output = conv_output.permute(0, 2, 1) # 1, 63,128
        return conv_output

class ClassifierNetwork(nn.Module):

    def __init__(self, cfgs):
        super(ClassifierNetwork, self).__init__()
        self.cfgs = cfgs

        self.text_transformation = nn.Linear(self.cfgs.text_dim, self.cfgs.hidden_dim)
        self.audio_transformation = nn.Linear(self.cfgs.audio_dim, self.cfgs.hidden_dim)
        self.video_transformation = nn.Linear(self.cfgs.video_dim, self.cfgs.hidden_dim)

        # text encoder initialization
        if self.cfgs.text_encoder == "cnn_rnn":
            self.text_encoder = CNN_RNNEncoder(self.cfgs)
        elif self.cfgs.text_encoder == "transformer":
            self.text_encoder = TransformerEncoder(self.cfgs)
        elif self.cfgs.text_encoder == "cnn":
            self.text_encoder = CNNEncoder(self.cfgs)
        elif self.cfgs.text_encoder == "lstm": # by defualt using simple lstm encoder
            self.text_encoder = LSTMContex(self.cfgs)
        
        # audio encoder initialization
        if self.cfgs.audio_encoder == "cnn_rnn":
            self.audio_encoder = CNN_RNNEncoder(self.cfgs)
        elif self.cfgs.audio_encoder == "transformer":
            self.audio_encoder = TransformerEncoder(self.cfgs)
        elif self.cfgs.audio_encoder == "cnn":
            self.audio_encoder = CNNEncoder(self.cfgs)
        elif self.cfgs.audio_encoder == "lstm": # by defualt using simple lstm encoder
            self.audio_encoder = LSTMContex(self.cfgs)
        
        # video encoder initialization
        if self.cfgs.video_encoder == "cnn_rnn":
            self.video_encoder = CNN_RNNEncoder(self.cfgs)
        elif self.cfgs.video_encoder == "transformer":
            self.video_encoder = TransformerEncoder(self.cfgs)
        elif self.cfgs.video_encoder == "cnn":
            self.video_encoder = CNNEncoder(self.cfgs)
        elif self.cfgs.video_encoder == "lstm": # by defualt using simple lstm encoder
            self.video_encoder = LSTMContex(self.cfgs)
        
        self.fc = nn.Linear(3*self.cfgs.hidden_dim, 1)
        self.cls = nn.Sigmoid()
        
    def forward(self, text_feature,audio_feautre, video_feature, q_mask):
        
        text_feature = self.text_transformation(text_feature)
        audio_feature = self.audio_transformation(audio_feautre)
        video_feature = self.video_transformation(video_feature)
        
        # check whether one of the encoders is transformer
        # else pass the feature to its corresponding categorical encoder
        if self.cfgs.text_encoder == "transformer":
            text_feat = self.transformer_input(text_feature, q_mask, self.text_encoder)
        else:
            text_feat = self.text_encoder(text_feature, q_mask)
        if self.cfgs.audio_encoder == "transformer":
            audio_feat = self.transformer_input(audio_feature, q_mask, self.audio_encoder)
        else:
            audio_feat = self.audio_encoder(audio_feature, q_mask)
        if self.cfgs.video_encoder == "transformer":
            video_feat = self.transformer_input(video_feature, q_mask, self.video_encoder)
        else:
            video_feat = self.video_encoder(video_feature, q_mask)
               
        concate = torch.cat((text_feat, audio_feat, video_feat), 2)

        dense_rep = self.fc(concate)
        predicted = self.cls(dense_rep)
        # predicted = dense_rep

        return predicted
    
    def transformer_input(self, query_features, q_mask, encoder):
        query_features = query_features.transpose(0,1)
        query_mask = torch.zeros(q_mask.size()[0],q_mask.size()[1], dtype=torch.bool)
        for i in range(q_mask.size()[0]):
            for j in range(q_mask.size()[1]):
                if q_mask[i][j] == 0:
                    query_mask[i][j] = True
                else:
                    query_mask[i][j] = False
        query_mask = query_mask
        query_features = encoder(query_features, query_mask)
        query_features = query_features.transpose(0,1) # 1, 63, 128
        
        return query_features


                                                         