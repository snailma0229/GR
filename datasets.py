from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

class Dataset_Kuai(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        alist = self.features[idx, :]

        usr_feas = torch.tensor(alist[:19])
        item_feas = torch.tensor(alist[19:24].tolist() + [alist[-1]])

        usr_item_feas = torch.cat([usr_feas, item_feas], dim=0).int()
        umsk = torch.tensor(alist[24:43], dtype=torch.float32)
        imsk = torch.tensor(alist[43:48], dtype=torch.float32)
        play_time = torch.tensor([alist[-5]], dtype=torch.float32)
        labels = torch.tensor([self.labels[idx]], dtype=torch.float32).squeeze(0)
        return usr_item_feas, umsk, imsk, play_time, labels

def collate_fn(batch):

    features, qmsks, imsks, play_times, labels = zip(*batch)

    features = torch.stack(features)
    qmsks = torch.stack(qmsks)
    imsks = torch.stack(imsks)
    play_times = torch.stack(play_times)
    
    labels_padded = pad_sequence([l.clone().detach() for l in labels],
                             batch_first=True, padding_value=0)
    masks = pad_sequence([torch.ones(len(l), dtype=torch.long) for l in labels],
                            batch_first=True, padding_value=0)

    return features, qmsks, imsks, play_times, labels_padded, masks