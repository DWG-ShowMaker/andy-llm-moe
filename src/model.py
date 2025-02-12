import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) 层

    Args:
        input_dim (int): 输入特征的维度。
        output_dim (int): 输出特征的维度。
        num_experts (int): 专家数量。
        hidden_dim (int): 每个专家隐藏层的维度。
    """
    def __init__(self, input_dim, output_dim, num_experts=4, hidden_dim=64):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        
        # 定义每个专家（简单的两层全连接网络，带有 ReLU 激活）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_experts)
        ])
        
        # 定义门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量，形状 (batch_size, input_dim)
        
        Returns:
            Tensor: 来自各专家加权和的输出，形状 (batch_size, output_dim)
        """
        # 计算门控网络得分，并使用 softmax 标准化
        gate_scores = self.gate(x)  # (batch_size, num_experts)
        gate_probs = F.softmax(gate_scores, dim=-1)  # (batch_size, num_experts)
        
        # 计算每个专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每个输出形状: (batch_size, output_dim)
        
        # 将专家输出堆叠, 形状: (batch_size, num_experts, output_dim)
        expert_stack = torch.stack(expert_outputs, dim=1)
        
        # 扩展 gate_probs 以匹配专家输出维度
        gate_probs = gate_probs.unsqueeze(-1)  # (batch_size, num_experts, 1)
        
        # 计算各专家输出的加权和
        output = torch.sum(gate_probs * expert_stack, dim=1)  # (batch_size, output_dim)
        return output


class ChineseMoEModel(nn.Module):
    """
    小型中文模型，结合嵌入层、MoE 层和分类器。

    结构说明:
      - 嵌入层: 将输入的词索引转换成词向量；
      - 平均池化: 计算句子中所有词向量的平均，得到句子级表示；
      - MoE层: 对句子表示进行特征变换；
      - 分类器: 最后通过全连接层输出分类结果。

    Args:
        vocab_size (int): 词汇表大小。
        embed_dim (int): 词向量维度。
        num_classes (int): 分类数目。
        moe_num_experts (int): MoE 层专家数量。
        moe_hidden_dim (int): 每个专家隐藏层的维度。
    """
    def __init__(self, vocab_size, embed_dim=128, num_classes=2, moe_num_experts=4, moe_hidden_dim=64):
        super(ChineseMoEModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # MoE 层，输入和输出均为 embed_dim
        self.moe = MoE(input_dim=embed_dim, output_dim=embed_dim,
                       num_experts=moe_num_experts, hidden_dim=moe_hidden_dim)
        
        # 分类器：将 MoE 层输出映射到分类结果
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入句子张量，形状 (batch_size, seq_len)

        Returns:
            Tensor: 分类 logits，形状 (batch_size, num_classes)
        """
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # 使用 mask 排除 padding 部分，计算平均池化后的句子表示
        mask = (x != 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
        summed = torch.sum(embedded * mask, dim=1)  # (batch_size, embed_dim)
        lengths = torch.clamp(torch.sum(mask, dim=1), min=1e-6)
        pooled = summed / lengths  # (batch_size, embed_dim)
        
        moe_out = self.moe(pooled)            # (batch_size, embed_dim)
        logits = self.classifier(moe_out)       # (batch_size, num_classes)
        return logits