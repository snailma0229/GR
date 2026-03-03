import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, use_embedding_mixup, max_values_per_column, feat_dim, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.use_embedding_mixup = use_embedding_mixup

        self.user_id_embedding = nn.Embedding(int(max_values_per_column[0])+1, feat_dim)
        self.user_fea_embedding = nn.ModuleList([
            nn.Embedding(int(max_values_per_column[i])+1, feat_dim) for i in range(1, 19)
        ])
        self.item_id_embedding = nn.Embedding(int(max_values_per_column[19])+1, feat_dim)
        self.item_fea_embedding = nn.Embedding(int(max(max_values_per_column[20:24]))+1, feat_dim)
        self.item_duration_embedding = nn.Embedding(int(max_values_per_column[-1])+1, feat_dim)

        self.device = device
    
    def make_trg_mask(self, trg, trg_pad_mask):
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask.unsqueeze(1) & trg_sub_mask.unsqueeze(0)

        return trg_mask.unsqueeze(1)
    
    def get_encoder_result(self, x, qmsk, imsk):

        usr_id = self.user_id_embedding(x[:, 0].unsqueeze(1))
        usr_fea = torch.cat([self.user_fea_embedding[id](x[:, i].unsqueeze(1)) for id, i in enumerate(range(1,19))], dim=1)
        item_id = self.item_id_embedding(x[:, 19].unsqueeze(1))
        item_fea = torch.cat([self.item_fea_embedding(x[:, i].unsqueeze(1)) for i in range(20,24)], dim=1)
        item_duration = self.item_duration_embedding(x[:, 24].unsqueeze(1))

        usr_fea = torch.cat([usr_id, usr_fea], dim=1)
        item_fea = torch.cat([item_id, item_fea], dim=1)

        qmsk = qmsk.unsqueeze(-1)
        imsk = imsk.unsqueeze(-1)
        
        query_embeddings = usr_fea * qmsk
        item_embeddings = item_fea * imsk
        item_embeddings = torch.cat([item_embeddings, item_duration], dim=1)
        
        query_sum = query_embeddings.sum(1) / qmsk.sum(1)
        item_sum = item_embeddings.sum(1) / (imsk.sum(1) + 1)
        src_input = torch.cat([query_sum, item_sum], dim=1)

        enc_src = self.encoder(src_input).transpose(0,1) 
        
        src_mask = torch.ones(qmsk.shape[0], item_duration.shape[1]).to(self.device).unsqueeze(1).unsqueeze(2)

        return enc_src, src_mask
    
    def forward(self, x, qmsk, imsk, trg, true_label_mask, teacher_force_ratio):

        enc_src, src_mask = self.get_encoder_result(x, qmsk, imsk)

        trg_mask = self.make_trg_mask(trg, true_label_mask)

        output, output_fea = self.decoder(trg, enc_src, trg_mask, src_mask)

        input_teacher = []
        teacher_force = [random.random() > teacher_force_ratio for _ in range(1, output.shape[1])]
        prob = output.argmax(dim=-1)

        for i in range(output.shape[1]):
            if teacher_force[i-1] and i != 0:
                if self.use_embedding_mixup:
                    input_teacher.append(output[:,i-1,:])
                else:
                    input_teacher.append(prob[:,i-1].unsqueeze(1))
            else:
                input_teacher.append(trg[:,i].unsqueeze(1))

        if not self.use_embedding_mixup:
            input_teacher = torch.cat(input_teacher, dim=1)

        output, _ = self.decoder.weight_forward(input_teacher, enc_src, trg_mask, src_mask, self.use_embedding_mixup)

        return output