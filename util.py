import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

def get_genral_vocab_dynamic_q(watch_ratio_all_for_vocab, q_start, q_end, q_decay_rate, epsilon):
    V = []
    cnt = []
    Y_sorted = np.sort(watch_ratio_all_for_vocab)[::-1]
    Y_updated = Y_sorted.copy()
    err = float('inf')
    q = q_start
    zi = float('1000')

    while err > epsilon and int(zi) != 0:
        zi = np.percentile(Y_updated, q * 100)
        if int(zi) == 0:
            break
        Y_updated = np.where(Y_updated < zi, Y_updated, Y_updated - zi)
        V.append(int(zi))
        cnt.append(Y_updated[Y_updated>=zi].shape[0])
        err = np.max(np.abs(Y_updated / (Y_sorted + 1e-10)))
        q = max(q_end, q * q_decay_rate)

    return V


def reduce_numbers(number, vocab):
    new_label = []
    remainder = number
    for value in vocab:
        while remainder >= value:
            remainder -= value
            new_label.append(value)
    return remainder, sorted(list(new_label), reverse=True)

def label_process(vocab, numeric_vocab, len_train, watch_ratio_all_for_vocab, device):
    w2i_vocab = {word: idx for idx, word in enumerate(vocab)}
    i2w_vocab = {idx: word for idx, word in enumerate(vocab)}

    i2v_vocab = {}
    for idx, _ in i2w_vocab.items():
        if idx == 2 or idx == 0 or idx == 1:
            i2v_vocab[idx] = 0
        else:
            i2v_vocab[idx] = vocab[idx]
    vocab_size = len(i2v_vocab)
    vocab_values = torch.empty(vocab_size, dtype=torch.float32).to(device)
    for idx, value in i2v_vocab.items():
        vocab_values[idx] = value

    durations = []
    for i in vocab:
        if not isinstance(i, str):
            durations.append(i)
        else:
            durations.append(0)
    durations = torch.tensor(durations, device=device)

    watch_ratio_labels = []
    for index, watch_ratio in enumerate(tqdm(watch_ratio_all_for_vocab.tolist(), desc='Label processing')):
        watch_ratio_all_for_vocab[index], new_label = reduce_numbers(watch_ratio, numeric_vocab)
        label = ['<sos>'] + new_label + ['<eos>']
        watch_ratio_labels.append([vocab.index(i) for i in label])
    assert sum(watch_ratio_all_for_vocab != 0) == 0

    watch_ratio_labels_train = watch_ratio_labels[:len_train]
    watch_ratio_labels_test = watch_ratio_labels[len_train:]

    return durations, vocab_values, watch_ratio_labels_train, watch_ratio_labels_test


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), seconds

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
