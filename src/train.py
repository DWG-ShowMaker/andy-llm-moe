import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ChineseMoEModel
from data_loader import ChineseDataset, collate_fn

def train(args):
    # 选择设备：优先使用 MPS（适用于 M1 芯片），其次检查 CUDA，最后使用 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 初始化数据集和数据加载器
    dataset = ChineseDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 初始化模型，获取词汇表大小并设置各层参数
    vocab_size = len(dataset.vocab)
    model = ChineseMoEModel(vocab_size=vocab_size, embed_dim=args.embed_dim, num_classes=args.num_classes,
                            moe_num_experts=args.moe_num_experts, moe_hidden_dim=args.moe_hidden_dim)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 模型训练
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # 输出形状: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), args.save_path)
    print(f"模型已保存到 {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a small Chinese model with MoE architecture")
    parser.add_argument('--epochs', type=int, default=10, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=16, help="批量大小")
    parser.add_argument('--lr', type=float, default=0.001, help="学习率")
    parser.add_argument('--embed_dim', type=int, default=128, help="嵌入层维度")
    parser.add_argument('--num_classes', type=int, default=2, help="分类数目")
    parser.add_argument('--moe_num_experts', type=int, default=4, help="MoE 专家数量")
    parser.add_argument('--moe_hidden_dim', type=int, default=64, help="MoE 专家隐藏层维度")
    parser.add_argument('--log_interval', type=int, default=1, help="日志打印间隔")
    parser.add_argument('--save_path', type=str, default="moe_model.pt", help="模型保存路径")
    
    args = parser.parse_args()
    train(args)