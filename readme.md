# 小型中文模型训练项目 - MoE 架构

## 概述

本项目使用 Mixture-of-Experts (MoE) 架构训练一个小型中文模型。该项目主要用于演示如何使用 MoE 模型进行中文自然语言处理任务，包含的功能模块有：

- 基于 PyTorch 实现的 MoE 模型（详见 `src/model.py`）；
- 简单的中文数据集及数据加载模块（`src/data_loader.py`）；
- 训练脚本（`src/train.py`），包括训练循环、超参数配置及模型保存；
- 推理脚本（`src/inference.py`），加载训练后的模型并对新的中文文本进行推理，同时使用标签映射输出更直观的文本描述。

## 项目结构

```text
├── readme.md               # 项目文档及使用说明
├── requirements.txt        # 项目依赖
└── src
    ├── model.py            # MoE 模型及小型中文模型定义
    ├── data_loader.py      # 中文数据集及数据加载器
    ├── train.py            # 模型训练脚本
    └── inference.py        # 模型推理脚本
```

## 使用方法

### 环境安装

- 推荐使用 Python 3.8 及以上版本。
- 安装依赖（可直接执行）：
  ```bash
  pip install -r requirements.txt
  ```

### 数据准备

`src/data_loader.py` 内置一个简单示例中文数据集，用于演示训练流程。在实际应用中，你可以修改或扩展此模块以加载真实中文数据。

---

## 训练前准备

请按照以下步骤进行环境搭建和准备，然后再开始模型训练：

1. **克隆仓库**  
   ```bash
   git clone https://github.com/DWG-ShowMaker/andy-llm-moe.git
   cd andy-llm-moe
   ```

2. **创建并激活虚拟环境**  
   - **Linux/macOS：**
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - **Windows：**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **安装依赖**  
   对于 Apple M1 芯片，建议安装支持 MPS 的 PyTorch 版本以获得硬件加速。首先卸载旧版本的 torch 和 torchvision，然后安装支持 MPS 的版本：
   ```bash
   pip uninstall torch torchvision
   pip install torch==1.12.0 torchvision --pre
   ```
   接着安装其他依赖：
   ```bash
   pip install -r requirements.txt
   ```

4. **验证环境**  
   在 Python REPL 中执行以下代码，确认 MPS 后端可用：
   ```python
   import torch
   print(torch.backends.mps.is_available())  # 如果输出 True，则表示 MPS 可用
   ```

---

## 训练模型

使用 `src/train.py` 脚本开始训练模型，你可以通过命令行参数自定义训练轮数、批量大小、学习率等超参数。

**在 Apple M1 设备上运行示例：**
```bash
python src/train.py --epochs 10 --batch_size 16 --lr 0.001
```

训练过程中，程序会自动检测设备（优先使用 MPS，其次 CUDA，最后 CPU），并在终端输出每个 batch 的损失信息以及每个 epoch 的平均损失。训练结束后，模型参数会保存在默认路径 `moe_model.pt`（或你自定义的路径）。

---

## 模型推理使用

训练完成后，你可以使用 `src/inference.py` 脚本加载模型，并对新的中文文本进行推理。请确保推理时所使用的超参数（例如 `embed_dim`、`num_classes`、`moe_num_experts`、`moe_hidden_dim`）与训练时保持一致，以便正确加载模型和词汇表。

**在 Apple M1 设备上运行示例：**
```bash
python src/inference.py --text "我爱编程"
```

推理脚本将执行以下步骤：
- 加载训练时生成的词汇表（内置于 `ChineseDataset` 中）。
- 加载保存的模型文件 `moe_model.pt`。
- 将输入文本转换为 token 索引，经过模型推理后输出预测类别及类别对应的概率分布。
- 通过预定义的标签映射（例如将 0 映射为"负面"，1 映射为"正面"）输出更直观的结果。

---

## 对于 Apple M1 设备的额外说明

如果你使用 Apple M1 芯片的 Mac 设备，请注意以下几点以获得最佳性能：

1. **安装支持 MPS 的 PyTorch**  
   请确保安装 PyTorch 1.12 及以上版本。参考 [PyTorch 官网](https://pytorch.org) 获取最新安装命令：
   ```bash
   pip uninstall torch torchvision
   pip install torch==1.12.0 torchvision --pre
   ```

2. **训练模型**  
   训练时程序会自动检测 MPS 并使用硬件加速。你可以使用常规命令启动训练：
   ```bash
   python src/train.py --epochs 10 --batch_size 16 --lr 0.001
   ```
   在训练日志中你可以看到所使用的设备类型（MPS、CUDA 或 CPU）。

3. **推理使用**  
   推理时同样会自动检测 MPS 后端，确保你按照上述步骤正确安装了 PyTorch。

---

## 模型说明

- **MoE 模块**：实现了多个专家和门控网络的 MoE 层。每个专家采用简单的全连接结构，门控网络根据输入特征分配权重。
- **小型中文模型**：由嵌入层、MoE 层和分类器组成。首先对输入的中文句子（逐字符）生成嵌入，再通过平均池化获得句子级表示，随后经 MoE 层进行特征变换，最后由分类器输出预测结果。

---

## 超参数

- `epochs`：训练的轮数
- `batch_size`：每个批次的样本数量
- `lr`：学习率
- `embed_dim`：词嵌入的维度
- `num_classes`：分类类别数（例如：0 表示负面，1 表示正面）
- `moe_num_experts`：MoE 层中专家的数量
- `moe_hidden_dim`：每个专家隐藏层的维度

---

## 模型训练与监控

- **监控指标**：训练过程中会实时输出每个 batch 的损失值以及每个 epoch 的平均损失，方便用户监控模型训练情况。
- **错误监控**：代码内含基础异常捕捉与日志输出，可根据需要进一步完善监控手段。

---

## 总结

本项目提供了一个基于 MoE 架构的中文模型训练示例，旨在帮助研究者和开发者快速了解 MoE 的原理与实现方式，并为大家提供一个开源的学习资源。你可以在此基础上扩展数据集、调整网络结构或添加更多功能。如果在使用过程中有任何问题或改进建议，欢迎反馈或提交 Pull Request。

---

## 项目背景

本项目致力于提供一个轻量级、中英文自然语言处理示例，通过 MoE 架构提升模型表达能力和泛化效果，尤其关注在专家网络中优化参数利用率和提高模型鲁棒性。

---

## 技术栈与依赖

- **Python**：3.8 及以上版本
- **PyTorch**：1.7 及以上版本（建议在 Apple M1 设备上使用支持 MPS 的 PyTorch 1.12 及以上版本）
- 其他依赖项请参考 [requirements.txt](requirements.txt)

---

## 如何贡献

请参考 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。我们欢迎任何形式的代码改进和建议！

---

## 联系方式

如有疑问或建议，请通过 [746144374@qq.com](mailto:746144374@qq.com) 与我们联系。

---

## 未来展望

本项目未来可能会加入以下特性：
- 数据预处理和增强工具
- 更多模型架构及对比实验
- 模型部署和实时推理支持

我们期待与社区共同完善和扩展此项目。
