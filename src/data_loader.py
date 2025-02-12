import torch
from torch.utils.data import Dataset

class ChineseDataset(Dataset):
    """
    简单的中文数据集，用于演示。

    数据格式为字典，包含 'text' 和 'label' 字段。
    内部自动构建词汇表（每个汉字为一个 token），并将文本转换为对应的索引列表。
    """
    def __init__(self):
        # 示例数据
        self.data = [
            {"text": "我爱编程", "label": 1},
            {"text": "天气糟糕", "label": 0},
            {"text": "今天心情不错", "label": 1},
            {"text": "真倒霉", "label": 0}
        ]
        
        # 初始化词汇表，预留索引 0 作为 padding token
        self.vocab = {"<pad>": 0}
        for item in self.data:
            for char in item["text"]:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
                    
        # 将文本转换为 token 索引
        self.samples = []
        for item in self.data:
            token_ids = [self.vocab[char] for char in item["text"]]
            self.samples.append((token_ids, item["label"]))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        token_ids, label = self.samples[idx]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    批处理函数：对输入样本进行 padding，使每个 batch 中的句子长度一致。

    Args:
        batch: List of tuples (token_tensor, label_tensor)

    Returns:
        Tuple: (padded_tokens, labels)，padded_tokens 的形状为 (batch_size, max_seq_len)
    """
    # 找到 batch 中最大的句子长度
    max_len = max([item[0].size(0) for item in batch])
    
    padded_tokens = []
    labels = []
    for tokens, label in batch:
        pad_size = max_len - tokens.size(0)
        if pad_size > 0:
            padded = torch.cat([tokens, torch.zeros(pad_size, dtype=torch.long)])
        else:
            padded = tokens
        padded_tokens.append(padded.unsqueeze(0))
        labels.append(label.unsqueeze(0))
        
    padded_tokens = torch.cat(padded_tokens, dim=0)
    labels = torch.cat(labels, dim=0)
    return padded_tokens, labels