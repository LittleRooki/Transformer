import dataprocess
from Config import Config
import torch

def greedy_search(model, encoder_input, start):
    encoder_output = model.encoder(encoder_input)
    decoder_input = torch.zeros(1, 0).type_as(encoder_input.data)
    flag = False
    next_word = start
    while not flag:
        next_word_tensor = torch.tensor([[next_word]], dtype=encoder_input.dtype)
        decoder_input = torch.cat([decoder_input.to(Config.device),
                                   next_word_tensor.to(Config.device)], -1)
        decoder_output = model.decoder(decoder_input, encoder_input, encoder_output)
        projection = model.projection(decoder_output)
        pro = projection.squeeze(0).max(dim=-1, keepdim=False)[1]

        next_p = pro.data[-1]
        next_word = next_p
        if next_word == tgt_vocab["<eos>"]:
            flag = True

    greedy_predict = decoder_input[:, 1:]
    return greedy_predict


if __name__ == '__main__':
    sent = [['wir werden alle geboren . wir bringen kinder zur welt .'],
            ['wir durch@@ laufen initi@@ a@@ tions@@ ritu@@ ale .']]
    src_vocab = torch.load('de_vocab')
    tgt_vocab = torch.load('en_vocab')
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}

    tmp = sent[0][0].split()
    enc_input = []
    for i in range(len(tmp)):
        enc_input.append(src_vocab[tmp[i]])


    model = torch.load("Transformer_4.pkl")
    predict = greedy_search(model,
                            torch.tensor(enc_input).view(1, -1).to(Config.device),
                            start=tgt_vocab["<sos>"])
    tgt = [idx2word[n.item()] for n in predict.squeeze()]
    tgt = " ".join(tgt)
    print("source sentence:")
    print(sent[0][0])
    print("predict sentence:")
    print(tgt)














