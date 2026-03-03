import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from metrics import xauc_score

def evaluate(test_loader, model, window_size, use_embedding_mixup, durations, device, max_len = 20):
    
    model.eval()
    
    print("Evaluate the model on the test set...")

    gt_times_list = []
    all_pred_list = []
    all_ground_list = []
    with torch.no_grad():
        for X_batch, qmsk_batch, imsk_batch, play_times, Seq_label_batch, Seq_label_mask in test_loader:
            X_batch = X_batch.to(device)
            qmsk_batch = qmsk_batch.to(device)
            imsk_batch = imsk_batch.to(device)
            Ytime_batch = play_times.to(device)
            Seq_label_batch = Seq_label_batch.to(device=device, dtype=torch.long).transpose(0,1)
            Seq_label_mask = Seq_label_mask.to(device)

            enc_src, src_mask = model.get_encoder_result(X_batch, qmsk_batch, imsk_batch)

            input = Seq_label_batch[0, :].unsqueeze(1)        
            output_prob = []

            if use_embedding_mixup:
                input_list = []
                input_list.append(input)
                for _ in range(max_len - 1):
                    trg_pad_mask = torch.ones_like(input, dtype=torch.bool)
                    trg_mask = model.make_trg_mask(input, trg_pad_mask)

                    output, _ = model.decoder.weight_forward(input_list, enc_src, trg_mask, src_mask, use_embedding_mixup) # (batch_size, seq, vocab) / (batch_size, seq, dim)
                    next_word = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(-1) # # (batch_size, 1)
                    input = torch.cat([input, next_word], dim=1) # 这里的input就是为了来构建下三角矩阵的
                    input_list.append(output[:, -1,:]) # (batch_size, vocab)
                    prob = F.softmax(output[:, -1, :], dim=1)
                    output_prob.append(prob)
            else:
                for _ in range(max_len - 1):
                    trg_pad_mask = torch.ones_like(input, dtype=torch.bool)
                    trg_mask = model.make_trg_mask(input, trg_pad_mask)

                    output, _ = model.decoder(input, enc_src, trg_mask, src_mask) # batch, seq, vocab
                    prob = F.softmax(output[:, -1, :], dim=-1) # batch, vocab
                    next_word = torch.argmax(prob, dim=-1).unsqueeze(-1)  # batch, 1
                    input = torch.cat([input, next_word], dim=1)
                    output_prob.append(prob) 

            output_prob = [i.unsqueeze(1) for i in output_prob] # [(batch, vocab) * 19]
            outputs = torch.cat(output_prob, dim=1) # (batch, 19, vocab)
            max_indices = torch.tensor([i for i in input.tolist()], dtype=torch.int64).to(device).view(-1, max_len) # batch, 20
            gt_times_list = torch.tensor(Ytime_batch.squeeze().tolist(), dtype=torch.float64).to(device).view(-1)

            two_first_occurrences = (max_indices == 2).long().argmax(dim=1)
            row_length = max_indices.size(1)
            two_first_occurrences += ((max_indices == 2).sum(dim=1) == 0).long() * row_length
            range_indices = torch.arange(row_length).expand_as(max_indices).to(device)
            mask = range_indices <= two_first_occurrences.unsqueeze(1)

            # window
            argmax_indices = outputs.argmax(dim=-1) # batch, seq
            indices = torch.arange(outputs.shape[2], device=outputs.device).unsqueeze(0).unsqueeze(0).expand(outputs.shape[0], outputs.shape[1], -1)
            start_indices = argmax_indices.unsqueeze(-1) - window_size // 2 # batch, seq, 1
            end_indices = argmax_indices.unsqueeze(-1) + window_size // 2 + 1 # batch, seq, 1
            mask_batch = (indices >= start_indices) & (indices < end_indices) # batch, seq, vocab_num

            weighted_durations = torch.sum(outputs * mask_batch * durations.unsqueeze(0).unsqueeze(0), dim=-1)
            weighted_durations *= mask[:, 1:]
            pre_watch_times = torch.sum(weighted_durations, dim=1) / 1000

            preds = pre_watch_times.cpu().numpy()
            gts = gt_times_list.cpu().numpy()

            all_pred_list.append(preds)
            all_ground_list.append(gts)

    all_pred = np.concatenate(all_pred_list).reshape(-1)
    all_yy = np.concatenate(all_ground_list).reshape(-1)
    xauc = round(xauc_score(all_yy, all_pred), 4)  
    mae = round(np.mean(np.abs(all_pred - all_yy)), 4)
    return xauc, mae