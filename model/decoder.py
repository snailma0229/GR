import torch
import torch.nn as nn
from model.attention import PositionwiseFeedForward, MultiHeadAttention
import torch.nn.functional as F

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, drop_prob):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        return self.drop_out(tok_emb)


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

        self.tok_emb = nn.Embedding(dec_voc_size, d_model)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.output_dim = dec_voc_size

    def forward(self, trg, enc_src, trg_mask, src_mask):

        trg = self.drop_out(self.tok_emb(trg))

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)
        return output, trg

    def weight_forward(self, trg, enc_src, trg_mask, src_mask, use_embedding_mixup):

        if not use_embedding_mixup:
            trg = self.drop_out(self.tok_emb(trg))
        else:
            embeddeding = self.tok_emb.weight # (vocab, dim)
            embedded_list = []
            for i in trg:
                if i.shape[-1] == self.output_dim:
                    softmax_input = F.softmax(i, dim=-1) # (batch_size, vocab) 
                    embedded_item = self.drop_out(torch.einsum('bv,ve->be', softmax_input, embeddeding)).unsqueeze(1) # batch, 1, dim
                    embedded_list.append(embedded_item)
                else:
                    embedded_list.append(self.drop_out(self.tok_emb(i)))
            trg = torch.cat(embedded_list, dim=1)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)
        return output, trg
    

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x