import torch
import torch.nn.functional as F
import argparse

from model import ChineseMoEModel
from data_loader import ChineseDataset

def load_model(model_path, device, vocab_size, embed_dim=128, num_classes=2, moe_num_experts=4, moe_hidden_dim=64):
    """
    加载模型并设置为评估模式
    """
    model = ChineseMoEModel(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes,
                            moe_num_experts=moe_num_experts, moe_hidden_dim=moe_hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def text_to_tensor(text, vocab):
    """
    将输入中文文本转换为模型输入的 token 索引张量。
    
    Args:
        text (str): 输入的中文文本。
        vocab (dict): 字符到索引的映射字典。
    
    Returns:
        Tensor: 形状为 (1, seq_len) 的 LongTensor。
    """
    # 对于未知字符，映射为 0（通常对应于 <pad> 或 <unk>）
    tokens = [vocab.get(char, 0) for char in text]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

def main(args):
    # 利用 ChineseDataset 提取训练时自动生成的词汇表
    dataset = ChineseDataset()
    vocab = dataset.vocab
    vocab_size = len(vocab)
    
    # 选择设备，优先使用 MPS（适用于 Apple M1），其次 CUDA，最后使用 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # 加载模型（根据参数指定的超参数需与训练时保持一致）
    model = load_model(args.model_path, device, vocab_size, args.embed_dim, args.num_classes, args.moe_num_experts, args.moe_hidden_dim)
    
    # 预处理输入文本
    inputs = text_to_tensor(args.text, vocab).to(device)

    # 推理
    with torch.no_grad():
        logits = model(inputs)
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        
    # 假设这是一个二分类任务，定义类别映射（请根据实际任务修改）
    label_map = {0: "负面", 1: "正面"}
    predicted_label_text = label_map.get(predicted_class, "未知")

    print(f"输入文本: {args.text}")
    print(f"预测类别: {predicted_class}，标签: {predicted_label_text}")
    print(f"类别概率: {probabilities}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for ChineseMoEModel")
    parser.add_argument('--model_path', type=str, default="moe_model.pt", help="训练后保存的模型文件路径")
    parser.add_argument('--text', type=str, required=True, help="待推理的中文文本")
    parser.add_argument('--embed_dim', type=int, default=128, help="嵌入层维度")
    parser.add_argument('--num_classes', type=int, default=2, help="分类数目")
    parser.add_argument('--moe_num_experts', type=int, default=4, help="MoE 专家数量")
    parser.add_argument('--moe_hidden_dim', type=int, default=64, help="MoE 专家隐藏层维度")
    args = parser.parse_args()
    main(args) 