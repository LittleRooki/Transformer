import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Config import Config


def padding_mask(seq_q, seq_k):
    ## 计算q，k得分之后的mask矩阵 seq_q:[batch_size, seq_len],seq_k:[batch_size, seq_len]
    mask = seq_k.data.eq(0)
    mask = mask.unsqueeze(1) #[batch_size, 1, len_k]
    mask = mask.expand(seq_k.size(0), seq_q.size(1), seq_k.size(1))#[batch_size, len_q, len_k]
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model=Config.d_model,
                 n_heads=Config.n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.Weight_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.Weight_K = nn.Linear(d_model, self.d_k * n_heads)
        self.Weight_V = nn.Linear(d_model, self.d_v * n_heads)
        self.fc = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, input_Q, input_K, input_V, att_mask):
        #input_Q:[batch_size, len_q, d_model], att_mask:[batch_size, seq_len, seq_len]
        #spilt to n heads,Q[batch_size, n_heads, len_q, d_k]
        Q = self.Weight_Q(input_Q).view(input_Q.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Weight_K(input_K).view(input_Q.size(0), -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Weight_V(input_V).view(input_Q.size(0), -1, self.n_heads, self.d_v).transpose(1, 2)

        att_mask = att_mask.unsqueeze(1).repeat(1, self.n_heads, 1 ,1)
        # att_output[batch_size, n_heads, len_q, d_v]
        att_output = Scaled_Dot_Product_Attention()(Q, K, V, att_mask)
        #concat n heads [batch_size, len_q, n_heads * d_v]
        att_output = att_output.transpose(1, 2).reshape(input_Q.size(0), -1, self.n_heads * self.d_v)
        #need a projection before into ffn
        att_output = self.fc(att_output) #[batch_size, len_q, d_model]
        return att_output


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()


    def forward(self, Q, K, V, mask):
        #Q:[batch_size, n_heads, len_q, d_k]
        # Q * K = score [batch_size, b_heads, len_q, len_k]
        score = torch.matmul(Q, K.transpose(-1, -2))
        score = score / np.sqrt(Q.size(-1)) # /sqrt(d_k)
        ## mask before Softmax
        score.masked_fill_(mask, -1e9)

        att = F.softmax(score, dim=-1) # probability distribution
        z = torch.matmul(att, V)#[batch_size, n_heads, len_q, d_v]
        return z


class PoswiseFeedwardNet(nn.Module):
    def __init__(self,
                 d_model=Config.d_model,
                 d_ffn=Config.d_ffn):
        super(PoswiseFeedwardNet, self).__init__()
        #feature extra
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ffn), #expand ->d_ffn
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)  #scale ->d_model
        )

    def forward(self, ffn_input):
        ffn_output = self.fc(ffn_input) #[batch_size, seq_len, d_model]
        return ffn_output


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 device=Config.device,
                 d_model=Config.d_model):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size).to(device)
        self.decoder = Decoder(tgt_vocab_size).to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size).to(device)
        self.init_params()


    def init_params(self):
        #init with xavier_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, encoder_input, decoder_input):
        # input [batch_size, src_length/tgt_length],

        encoder_output = self.encoder(encoder_input) #[batch_size, src_length, d_model]
        decoder_output = self.decoder(decoder_input,
                                      encoder_input,
                                      encoder_output)
        #output make a projection
        logit = self.projection(decoder_output)#[batch_size, tgt_len, tgt_vocab_size]
        output = logit.view(-1, logit.size(-1))#[batch_size*tgt_len, tgt_vocab_size]
        return output

class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 layers=Config.layers,
                 d_model=Config.d_model):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.encoder_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, encoder_input):
        #encoder_input:[batch_size, src_length]

        #word embedding + position embedding
        encoder_output = self.encoder_emb(encoder_input) * math.sqrt(self.d_model)#[batch_size, src_len, d_model]
        encoder_output = self.pos_emb(encoder_output)

        #[batch_size, src_len, src_len]
        encoder_mask = padding_mask(encoder_input, encoder_input)
        for layer in self.layers:
            encoder_output = layer(encoder_output, encoder_mask)

        encoder_output = self.norm(encoder_output)
        return encoder_output

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=Config.d_model,
                 dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadAttention()
        self.ffn = PoswiseFeedwardNet()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, encoder_input, encoder_mask):
        #encoder_input:[batch_size, src_len, d_model],encoder_mask[batch_size, src_len, src_len]
        encoder_input_norm = self.attn_norm(encoder_input)

        encoder_attention_output = self.attn(encoder_input_norm,
                                   encoder_input_norm,
                                   encoder_input_norm,
                                   encoder_mask)
        encoder_attention_output = self.dropout(encoder_attention_output)

        encoder_attention_output = encoder_input + encoder_attention_output

        ffn_input = self.ffn_norm(encoder_attention_output)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout(ffn_output)

        encoder_output = encoder_attention_output + ffn_output
        return encoder_output


class Decoder(nn.Module):
    def __init__(self,
                 tgt_vocab_size,
                 d_model=Config.d_model,
                 layers=Config.layers):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.decoder_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(layers)])

    def forward(self, decoder_input, encoder_input, encoder_output):
        decoder_output = self.decoder_emb(decoder_input) * math.sqrt(self.d_model) #[batch_size, tgt_len, d_model]
        decoder_output = self.pos_emb(decoder_output)

        #decoder_mask:[batch_size, tgt_len, tgt_len]
        decoder_mask = padding_mask(decoder_input, decoder_input)

        #future attention mask [batch_size, tgt_len, tgt_len]
        future_mask = future_att_mask(decoder_input)

        #add two mask
        decoder_att_mask = decoder_mask + future_mask
        decoder_att_mask = torch.gt(decoder_att_mask, 0)

        #dec_enc_att_mask[batch_size, tgt_len, src_len]
        decoder_encoder_att_mask = padding_mask(decoder_input, encoder_input)

        for layer in self.layers:
            decoder_output = layer(decoder_output,
                                   encoder_output,
                                   decoder_att_mask,
                                   decoder_encoder_att_mask)
        return decoder_output

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model=Config.d_model,
                 dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.decoder_att = MultiHeadAttention()
        self.decoder_encoder_att = MultiHeadAttention()
        self.ffn = PoswiseFeedwardNet()
        self.dec_att_norm = nn.LayerNorm(d_model)
        self.dec_enc_att_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                decoder_input,
                encoder_output,
                decoder_att_mask,
                decoder_encoder_att_mask):
        #dec_att
        decoder_input_norm = self.dec_att_norm(decoder_input)

        decoder_att_output = self.decoder_att(decoder_input_norm,
                                              decoder_input_norm,
                                              decoder_input_norm,
                                              decoder_att_mask)
        decoder_att_output = self.dropout(decoder_att_output)
        #add
        decoder_att_output = decoder_input + decoder_att_output

        # dec_enc_att
        decoder_encoder_input = self.dec_enc_att_norm(decoder_att_output)
        decoder_encoder_att_output = self.decoder_encoder_att(decoder_encoder_input,
                                                              encoder_output,
                                                              encoder_output,
                                                              decoder_encoder_att_mask)
        decoder_encoder_att_output = self.dropout(decoder_encoder_att_output)
        #add
        decoder_encoder_att_output = decoder_att_output + decoder_encoder_att_output

        #ffn
        ffn_input = self.ffn_norm(decoder_encoder_att_output)
        ffn_output = self.ffn(ffn_input)

        decoder_output = decoder_encoder_att_output + ffn_output
        return decoder_output




def future_att_mask(seq):
    #seq:[batch_size, tgt_len]
    att_shape = [seq.size(0), seq.size(1), seq.size(1)]
    future_mask = np.triu(np.ones(att_shape), k=1)
    future_mask = torch.from_numpy(future_mask).byte()#[batch_size, tgt_len, tgt_len]
    return future_mask


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 max_len=2000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # pe = pe.unsqueeze(0).transpose(0, 1)#[max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

