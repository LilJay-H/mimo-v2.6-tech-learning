# 04｜动手实验：从第一次训练到认识 RL

**执行状态：本文是经过上游代码阅读后制定的实验方案，不是已经在你的 Windows/GPU 环境上跑通的记录。**不要求一次完成全部实验。第一次只做实验 0、1；后面的实验在前一阶段解释得清楚后再进入。

## 实验 0：只验证环境，不下载大模型

建立单独的学习目录和虚拟环境，避免修改其他 AI 工具的 Python 环境。PyTorch 安装命令使用 [官方安装选择器](https://pytorch.org/get-started/locally/)生成；首次可以选择 CPU，确认代码正常后再启用 GPU。不要照抄旧教程的 CUDA 或 nightly 版本。

Windows PowerShell 示例：

```powershell
mkdir gpt-learning
cd gpt-learning
python -m venv .venv
# 使用官方选择器给出的安装参数，运行 .venv\Scripts\python.exe -m pip install ...
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.rand(2,3)); print('CUDA:', torch.cuda.is_available())"
```

Linux/macOS 对应的虚拟环境解释器是 `.venv/bin/python`。下面出现 `python` 时，指你选择好的虚拟环境解释器，不是另一个全局环境。

已有 NVIDIA GPU 时，再运行：

```text
nvidia-smi
```

记录 GPU 名称、显存容量、驱动版本、RAM、操作系统和 PyTorch 版本。`torch.cuda.is_available()` 为真后，再用一个小张量在 CUDA 上实际计算；不能只因系统显示有显卡就认定训练环境可用。

**通过标准：**能运行张量计算，能说清代码当前用的是 CPU 还是 GPU。此时还没有训练模型。

## 实验 1：训练一个很小的 GPT

### 1.1 取得教材

把教学仓库放在本学习仓库旁边，避免误将外部仓库和权重一起提交：

```text
git clone https://github.com/karpathy/ng-video-lecture.git
cd ng-video-lecture
git rev-parse HEAD
```

阅读 `gpt.py` 与 `input.txt`。上游来源：[仓库](https://github.com/karpathy/ng-video-lecture) · [gpt.py](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)。保存读到的提交 SHA；若上游代码结构变化，先对照，不要盲目按行号修改。

### 1.2 把首轮配置缩小

将 `gpt.py` 顶部对应变量改成：

```python
batch_size = 8
block_size = 64
max_iters = 300
eval_interval = 100
learning_rate = 1e-3
device = 'cpu'  # 第一轮先去掉 GPU 兼容性这个变量
eval_iters = 10
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.0
```

这是**本学习方案的缩小配置，不是作者默认配置，也不是最优超参数**。选择它是为了先看清循环。CPU 线程过多时可以试着设置 `torch.set_num_threads(2)`，但要测量，不把它当成必然提速。

首轮仅做功能验证。确认流程后，可把 `max_iters` 增加到 1000 或更多，观察变化；每次先测短运行，避免凭模型名字猜训练时长。

### 1.3 先预测，再执行

运行前写下：初始输出应该像什么？loss 为什么不是零？300 次参数更新后，预计哪些现象可能改变？然后运行：

```text
python gpt.py
```

保留训练/验证 loss 和最终文本。小模型输出混乱并不自动说明程序错误；还要看数值、数据与更新是否符合预期。

### 1.4 给训练加一个最小的保存/加载闭环

原教材没有完整 checkpoint 工作流。以下代码是**可选的教学修改**：保留原文件的导入、数据准备、`get_batch`、`estimate_loss` 和各个模型类；把从 `model = GPTLanguageModel()` 开始直到文件末尾的执行段替换为下列内容。代码依赖原文件前面定义的变量，**不是可独立运行的新文件**。

```python
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'sample'])
parser.add_argument('checkpoint', type=Path)
args = parser.parse_args()

# 模型结构必须与保存时一致。学习率和训练步数不是结构参数。
config = dict(block_size=block_size, n_embd=n_embd,
              n_head=n_head, n_layer=n_layer, dropout=dropout)
model = GPTLanguageModel().to(device)

if args.mode == 'train':
    if args.checkpoint.exists():
        raise FileExistsError('请换一个实验文件名，避免覆盖已有模型。')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(max_iters):
        if step % eval_interval == 0:
            metrics = estimate_loss()
            print(step, {k: float(v) for k, v in metrics.items()})
        x, y = get_batch('train')
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('final', {k: float(v) for k, v in estimate_loss().items()})
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    config=config, chars=chars, steps=max_iters),
               args.checkpoint)
else:
    # 只加载自己生成并信任的文件；不要加载不明来源的 checkpoint。
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if saved['config'] != config or saved['chars'] != chars:
        raise ValueError('模型配置或字符表不同，请恢复训练时配置与输入文本。')
    model.load_state_dict(saved['model'])
    print('loaded; optimizer updates in this run: 0')

model.eval()
torch.manual_seed(42)
with torch.no_grad():
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    ids = model.generate(start, max_new_tokens=200)[0].tolist()
    print(decode(ids))
```

然后分别运行：

```text
python gpt.py train runs/baseline.pt
python gpt.py sample runs/baseline.pt
```

第二条命令启动了一个新进程，读取的是自己的 checkpoint，而不是重新训练。当前这个简化版本仍要求原输入文件存在并且字符表一致；把分词器、配置和模型定义拆成独立推理模块，可以作为后续重构练习。

这里保存了优化器状态，但**尚未实现 resume 训练模式**；精确恢复训练还要考虑随机数状态、数据读取位置等，不能把“重新加载生成”误称为“逐位一致续训”。参考 [PyTorch 保存/加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。

**通过标准：**有一份从随机参数训练得到的权重，关闭程序后能读取它生成文本，并能说明采样时没有发生 `optimizer.step()`。

## 实验 2：做三个能解释的对照，而不是盲目调参

复制 [实验记录模板](../templates/experiment-log.md)，每次实验使用独立 checkpoint 路径。

| 实验 | 只改变的主变量 | 保持一致的关键项 | 观察 |
|---|---|---|---|
| A：学习率 | 例如 `3e-4 / 1e-3 / 3e-3` | 数据、模型结构、步数、随机种子、评测 | loss 稳定性与验证结果 |
| B：上下文 | 例如 `32 / 64 / 128` | 模型宽度/层数；明确按 token 数或步数对齐 | 信息利用、吞吐和内存 |
| C：数据量 | 原训练集与更小训练子集 | 验证集不变、模型配置不变 | 过拟合及生成风格 |

上下文实验中，改变长度会改变每步 token 数；不能把“相同步数”同时说成“相同训练量”。小实验至少记录一个固定的比较口径。较正式的结论需要多个随机种子；单次演示只支持有限结论。

**通过标准：**不只是说哪个更好，而是说明比较条件、结果和可能的混杂因素。

## 实验 3：把预训练连接到 SFT 和聊天

选择 [nanochat](https://github.com/karpathy/nanochat)，先读 `runs/runcpu.sh`，不要直接启动 `runs/speedrun.sh`。本次读取的 CPU 脚本包含数据准备、分词、预训练、评估、SFT；官方明确把它作为低成本教学路径，而非强能力模型配方。

建议分开执行与检查各阶段：

```text
分词器：encode/decode 能否往返？
预训练：训练前后 loss 与样本如何变化？
基础模型评测：保存 SFT 前的结果。
SFT：训练文本如何变成 user/assistant 对话？loss mask 在哪里？
聊天：用同一组固定提示比较 SFT 前后，不临时挑最好看的例子。
```

第一次可以选择较短的模型和训练预算；但不仅要改训练步数，还要检查 tokenizer 训练数据、评测数据量、序列长度和 checkpoint 位置。完整 CPU 脚本也可能下载较多数据，不因为文件名里有 CPU 就认定它只有几 MB。

**通过标准：**能解释“继续预测文本”和“按对话答案训练”的区别；保留前后检查点与固定评测结果。不要求小模型获得实用聊天能力。

参考：[runcpu.sh](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) · [speedrun.sh](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh)。两者当前都**不自动完成 RL**。

## 实验 4：MiMo 的任务—轨迹—评分闭环（此阶段不训练权重）

### 4.1 先检查发布材料是否真的能启动

```text
git clone --branch mimo-oss https://github.com/XiaomiMiMo/mimoagent.git
cd mimoagent
git rev-parse HEAD
python -c "from pathlib import Path; p=Path('example_configs/swe_docker.yaml'); print('Docker quickstart config exists:', p.is_file())"
```

本次读取时，README 提到的 `swe_docker.yaml` 返回 404；记录到的分支提交为 `467f0a19016f0ac4d63b8d17a1f0da9ba07f232c`。**如果文件仍不存在，不要继续照搬 README 的运行命令，也不要将 Kubernetes 配置仅改一个字段就视为兼容替代。**先核对该版本的配置 schema、容器后端、数据适配器与上游修复。

README 声明需要 Python 3.12 与 uv。可以先做代码阅读和任务/评分器检查，将完整 Agent 启动留到配置通过核验后。[mimoagent README](https://github.com/XiaomiMiMo/mimoagent/tree/mimo-oss)

### 4.2 配置可用后，按下面的顺序验收

先运行一个任务的**参考解评分**，确认测试工具本身会判分；然后验证无修改或错误解会失败；最后才让模型自行尝试。仅选一个安全的软件任务，单 worker、单 rollout，限制步骤、输出长度、运行时间与 API 花费。

README 中的 DeepSWE 示例是外部任务集，不是小米公布的 7k+ 任务全集。使用参考解是为了测试评分链路，不能把它混进模型正式评测输入。

读完一条轨迹后，做如下标注：任务目标 → 模型输出 → 工具执行 → 环境反馈 → 下一次决策 → 终止 → 最终评分。注意测试通过是否来自实际产物，而不是模型自己声称成功。

### 4.3 安全与费用边界

只在隔离容器和临时工作目录中执行模型生成的操作；不要默认使用可以访问个人文件的 local 后端。不要挂载个人主目录、SSH 密钥或 Docker socket，不使用 privileged 容器。开始阶段避开漏洞利用类环境，选择无外部目标的普通代码任务即可。

API Key 放环境变量或未跟踪的本地配置，不提交到 GitHub。尤其要检查默认“0”是否表示无限制：本次读取的 `mini-swe-agent.yaml` 中 `cost_limit: 0.0` 表示不设上限，不适合直接复制为学习预算。

**通过标准：**能用一条具体轨迹解释模型、工具、环境和评分器各自做了什么；知道这次运行没有改变模型权重。

## 实验 5：真正的强化学习，先做小任务而非 9B 全量配方

这是进阶实验设计，不是已验证的一键命令。可以先读 [nanochat 的 RL 脚本](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py)，理解其流程；上游脚本以 GSM8K 为任务，并明确说明算法做了简化。

选择一个已有一定成功率的小模型；任务可以缩小为程序能判定的一位数加法或字符串变换。划出互不重叠的训练题、调参验证题与最终测试题。然后：

```text
测基线 → 每题采样多个答案 → 评分 → 检查组内奖励是否有差异
       → 计算训练信号 → 更新少量参数步骤
       → 固定设置重新测试 → 分析成功率、格式错误率、生成长度
```

本次所读 `chat_rl.py` 用 `num_samples // device_batch_size` 分批生成；做小配置时应保持 `num_samples` 为 `device_batch_size` 的整数倍，不要随意改成不能整除的组合。脚本加载 SFT checkpoint，而不是从未训练的随机模型开始。

**在更新前检查三件事：**正确答案是否被判正确；明显错误或格式投机是否被拒绝；模型是否并非全对或全错。如果组内全是同一奖励，简单相对奖励方法可能没有区分信号，这时应先调整任务难度、基础模型或采样，而不是盲目延长训练。

只改变提示词、筛选最佳答案、调用 API 或反复运行 Agent，都不自动构成 RL。判断是否真的训练，要看到可解释的 loss/梯度/参数更新，以及更新前后的同条件评测。

**通过标准：**不强求分数提升；要求能证明参数是否更新，并解释独立评测的结果。若换成 LoRA、小模型或自定义任务，应明确标为教学变体，不称为 MiMo 论文复现。

## 不要在第一轮做的事情

不要下载 Pro/Flash 权重；不要启动默认 64-GPU 配方；不要为了第一条 loss 曲线先装 Kubernetes；不要用大量云 GPU 排查尚未验证的评分器；不要把所有任务失败都归咎于模型能力；也不要让 AI 自动把课程重构成看不懂的复杂项目。
