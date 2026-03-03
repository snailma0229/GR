import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))
from model.transformer import Seq2Seq
from model.encoder import Encoder
from model.decoder import Decoder
from util import seconds_to_hms, init_weights, count_parameters, label_process, get_genral_vocab_dynamic_q
import numpy as np
import argparse
from datasets import collate_fn, Dataset_Kuai
import random

import copy
import time
from sklearn.preprocessing import LabelEncoder
from inference import evaluate


def parse_args():
    """Parse command-line arguments for model training configuration."""
    parser = argparse.ArgumentParser(description="Set up model parameters")

    # Transformer architecture
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_head', type=int, default=8, help='head num')
    parser.add_argument('--dec_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--ffn_dim', type=int, default=256, help='feedforward dimension')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--cls_weight', type=float, default=10.0, help='Class weight')
    parser.add_argument('--huber_weight', type=float, default=1.0, help='Huber loss weight')
    parser.add_argument('--log_dir', type=str, default='checkpoints/', help='Log directory')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')

    # Data paths
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data (.npy)')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data (.npy)')

    # Dynamic vocabulary parameters
    parser.add_argument('--q_start', type=float, default=0.9999)
    parser.add_argument('--q_end', type=float, default=0.9)
    parser.add_argument('--q_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1e-6)

    # Model variants
    parser.add_argument('--use_embedding_mixup', action='store_true')
    parser.add_argument('--use_curriculum_learning', action='store_true')
    parser.add_argument('--curriculum_learning_type', type=str, choices=['linear', 'exp', 'sigmoid'], default='linear')
    parser.add_argument('--curriculum_learning_decay', type=float, default=0.9999)
    parser.add_argument('--teacher_force_ratio', type=float, default=0.5)
    parser.add_argument('--window_size', type=int, default=20, help='Window size')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_and_preprocess_data(args):
    """Load feature arrays and encode duration features via LabelEncoder.

    Returns:
        train_feats_selected: Training features with appended encoded duration column.
        test_feats_selected:  Test features with appended encoded duration column.
        watch_ratio_all_for_vocab: Concatenated watch ratios scaled to integer (x1000).
        max_values_per_column: Per-column max values across the full dataset.
    """
    train_feats_selected = np.load(args.train_data)
    test_feats_selected = np.load(args.test_data)

    watch_ratio_train = train_feats_selected[:, -4]
    watch_ratio_test = test_feats_selected[:, -4]
    watch_ratio_all = np.concatenate((watch_ratio_train, watch_ratio_test), axis=0)
    watch_ratio_all_for_vocab = (watch_ratio_all * 1000).astype(int)

    # Encode duration feature with LabelEncoder
    print(train_feats_selected.shape)
    print(test_feats_selected.shape)
    train_durations = train_feats_selected[:, -3]
    test_durations = test_feats_selected[:, -3]
    durations = np.concatenate((train_durations, test_durations), axis=0)
    encoder = LabelEncoder()
    unique_sorted_durations = np.sort(np.unique(durations))
    encoder.fit(unique_sorted_durations)
    encoded_duration = encoder.transform(durations)
    encoded_train_duration = encoded_duration[:len(train_feats_selected)]
    encoded_test_duration = encoded_duration[len(train_feats_selected):]
    train_feats_selected = np.column_stack((train_feats_selected, encoded_train_duration))
    test_feats_selected = np.column_stack((test_feats_selected, encoded_test_duration))
    print(train_feats_selected.shape)
    print(test_feats_selected.shape)

    feats_all = np.concatenate((train_feats_selected, test_feats_selected), axis=0)
    max_values_per_column = np.amax(feats_all, axis=0)

    return train_feats_selected, test_feats_selected, watch_ratio_all_for_vocab, max_values_per_column


def build_vocab(args, watch_ratio_all_for_vocab):
    """Build the vocabulary using dynamic quantile-based decomposition.

    Returns:
        vocab: Full vocabulary list including special tokens and numeric entries.
        numeric_vocab: Numeric portion of the vocabulary.
    """
    special_tokens = ['<pad>', '<sos>', '<eos>']
    numeric_vocab = get_genral_vocab_dynamic_q(
        watch_ratio_all_for_vocab,
        q_start=args.q_start,
        q_end=args.q_end,
        q_decay_rate=args.q_decay_rate,
        epsilon=args.epsilon
    )
    vocab = special_tokens + numeric_vocab
    return vocab, numeric_vocab


def build_model(args, vocab_size, max_values_per_column, device):
    """Construct the Seq2Seq model with encoder and decoder.

    Returns:
        model: The initialized Seq2Seq model moved to the target device.
    """
    encoder = Encoder(input_dim=args.feat_dim, hidden_dim=args.hidden_dim, dropout_rate = args.dropout)

    decoder = Decoder(d_model=args.hidden_dim,
                      n_head=args.n_head,
                      ffn_hidden=args.ffn_dim,
                      dec_voc_size=vocab_size,
                      drop_prob=args.dropout,
                      n_layers=args.dec_layers)

    model = Seq2Seq(encoder, decoder, args.use_embedding_mixup, max_values_per_column, args.feat_dim, device).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(init_weights)
    return model


def build_optimizer(model, lr):
    """Build the Adam optimizer with separate weight decay for embedding layers.

    Returns:
        optimizer: Configured Adam optimizer.
    """
    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'embedding' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': embedding_params, 'weight_decay': 1e-4}, 
        {'params': other_params, 'weight_decay': 0} 
    ], lr=lr) 
    return optimizer


def train_one_epoch(epoch, num_epochs, model, train_loader, optimizer, criterion, args, durations, device):
    """Run one training epoch over the entire training set.

    Returns:
        epoch_loss: Accumulated loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    total_batch_count = 0

    for num_step, (X_batch, qmsk_batch, imsk_batch, play_times, Seq_label_batch, Seq_label_mask) in enumerate(train_loader, 1):
        X_batch = X_batch.to(device)
        qmsk_batch = qmsk_batch.to(device)
        imsk_batch = imsk_batch.to(device)
        Ytime_batch = play_times.to(device)
        Seq_label_batch = Seq_label_batch.to(device=device, dtype=torch.long)
        Seq_label_mask = Seq_label_mask.to(device)

        optimizer.zero_grad()

        if args.curriculum_learning_type == 'linear':
            teacher_force_ratio = max(0, args.teacher_force_ratio - args.curriculum_learning_decay * total_batch_count)
        elif args.curriculum_learning_type == 'exp':
            teacher_force_ratio = args.teacher_force_ratio * (args.curriculum_learning_decay ** total_batch_count)
        elif args.curriculum_learning_type == 'sigmoid':
            teacher_force_ratio = args.teacher_force_ratio * args.curriculum_learning_decay / (args.curriculum_learning_decay + math.exp(total_batch_count / args.curriculum_learning_decay))
        else:
            teacher_force_ratio = args.teacher_force_ratio

        output = model(X_batch, qmsk_batch, imsk_batch, Seq_label_batch[:,:-1], Seq_label_mask[:,:-1], teacher_force_ratio)
        output_softmax = F.softmax(output, dim=-1)

        output_dim = output.shape[-1]
        seq_len = output.shape[1]

        # Windowed soft-argmax for watch time prediction
        argmax_indices = output_softmax.argmax(dim=-1) 
        indices = torch.arange(output_dim, device=output.device).unsqueeze(0).unsqueeze(0).expand(output.shape[0], seq_len, -1)
        start_indices = argmax_indices.unsqueeze(-1) - args.window_size // 2
        end_indices = argmax_indices.unsqueeze(-1) + args.window_size // 2 + 1
        mask = (indices >= start_indices) & (indices < end_indices)

        weighted_durations = torch.sum(output_softmax * mask * durations.unsqueeze(0).unsqueeze(0), dim=-1)
        weighted_durations = weighted_durations * Seq_label_mask[:, 1:]
        pre_watch_times = torch.sum(weighted_durations, dim=1) / 1000
        
        logits = output.contiguous().view(-1, output_dim)
        trg = Seq_label_batch[:,1:].contiguous().view(-1)

        loss_ce = criterion(logits, trg)

        loss_huber = F.huber_loss(Ytime_batch.squeeze(), pre_watch_times, reduction='sum')

        loss = args.cls_weight * loss_ce + args.huber_weight * loss_huber

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        total_batch_count += 1

        if num_step == 1 or num_step % 5000 == 0:
            print("Epoch: {}/{}, Step: {}, cls Loss: {:.4f}, Huber Loss: {:.4f}".format(epoch, num_epochs, num_step, loss_ce.item(), loss_huber.item()))

    return epoch_loss


def main():
    """Main entry point for training and evaluation."""
    args = parse_args()
    config = vars(args)
    for key, value in config.items():
        print(f"{key}: {value}")

    # Prepare checkpoint directory
    log_path = os.path.join(args.log_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    save_path = os.path.join(args.log_dir, 'best_model.pth')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------- Data Loading & Preprocessing --------------------
    train_feats_selected, test_feats_selected, watch_ratio_all_for_vocab, max_values_per_column = \
        load_and_preprocess_data(args)

    # -------------------- Vocabulary Construction --------------------
    vocab, numeric_vocab = build_vocab(args, watch_ratio_all_for_vocab)

    durations, vocab_values, watch_ratio_labels_train, watch_ratio_labels_test = \
        label_process(vocab, numeric_vocab, len(train_feats_selected), watch_ratio_all_for_vocab, device)
    assert len(watch_ratio_labels_train) == train_feats_selected.shape[0]

    # -------------------- DataLoader --------------------
    train_dataset = Dataset_Kuai(train_feats_selected, watch_ratio_labels_train)
    test_dataset = Dataset_Kuai(test_feats_selected, watch_ratio_labels_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # -------------------- Model --------------------
    model = build_model(args, len(vocab), max_values_per_column, device)
    optimizer = build_optimizer(model, args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    best_mae = float('inf')
    best_xauc = 0
    best_model = None
    start_time = time.time()
    print("vocab: ", vocab)

    # -------------------- Training Loop --------------------
    for epoch in range(1, args.num_epochs+1):
        epoch_start = time.time()

        train_one_epoch(epoch, args.num_epochs, model, train_loader, optimizer, criterion, args, durations, device)

        print(f"{epoch}/{args.num_epochs} Optimization Finished!")

        # Evaluate with windowed soft-argmax (same strategy as training)
        xauc, mae = evaluate(test_loader, model, args.window_size, args.use_embedding_mixup, durations, device)
        print(f"{epoch}/{args.num_epochs} epoch XAUC is {xauc}")
        print(f"{epoch}/{args.num_epochs} epoch MAE is {mae}")

        # Track best model by MAE
        if mae < best_mae:
            best_mae = mae
            best_xauc = xauc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, save_path)
        
        epoch_end = time.time()
        elapsed_time = epoch_end - start_time
        remaining_epochs = args.num_epochs - epoch
        average_time_per_epoch = elapsed_time / epoch
        estimated_remaining_time = remaining_epochs * average_time_per_epoch
        elapsed_hours, elapsed_minutes, elapsed_seconds = seconds_to_hms(elapsed_time)
        remaining_hours, remaining_minutes, remaining_seconds = seconds_to_hms(estimated_remaining_time)
        print(f"Elapsed Time: {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds:.2f}s, Est Time: {remaining_hours}h {remaining_minutes}m {remaining_seconds:.2f}s")

        print(f"Best XAUC is {best_xauc} \nBest MAE is {best_mae}")


if __name__ == '__main__':
    main()