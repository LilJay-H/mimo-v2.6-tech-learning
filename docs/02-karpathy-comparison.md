# 02｜Karpathy 的哪个仓库？它和 MiMo 有何不同？

调研快照：2026-09-22。这里先辨认项目，再比较学习价值，避免把不同年份、不同目标的项目合并成一个“GPT 教程”。

## 一、你记忆中的 2024 年项目，最可能是哪个？

**最可能是 [`karpathy/build-nanogpt`](https://github.com/karpathy/build-nanogpt)**：它配套《Let's reproduce GPT-2 (124M)》视频，以逐步提交的代码讲解如何构建和训练 GPT-2 规模的模型。但仅凭“2024 年左右、自学模型训练”无法唯一确定；也可能指 nanoGPT 或 llm.c。以下将它们都定位清楚，而不是武断认定。[K1]

| 项目 | 主要定位 | 适合在学习路径中的位置 |
|---|---|---|
| [Zero to Hero](https://karpathy.ai/zero-to-hero.html)，含 micrograd、makemore | 从梯度、神经网络、张量和语言建模开始的课程 | 最先补最小基础 |
| [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | 《Let's build GPT》对应的字符级小 GPT，2023 年教学项目 | 第一套真正动手训练的 GPT |
| [build-nanogpt](https://github.com/karpathy/build-nanogpt) | 2024 年 GPT-2 124M 复现视频与渐进代码 | 了解标准架构及高效预训练，选修进阶 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 相对精简、可配置的 GPT 训练/微调实现 | 可参考其训练器；不要与视频仓库混称 |
| [llm.c](https://github.com/karpathy/llm.c) | C/CUDA 的语言模型训练实现，偏底层效率 | 对系统、GPU 内核感兴趣时再学 |
| [nanochat](https://github.com/karpathy/nanochat) | 2025 年开始的端到端小型聊天模型项目 | 想连接预训练、SFT、评测、聊天时使用 |

出处：[K0][K1][K2][K3][K4][K5]。**这些项目不需要全部做完。**它们不是一串必须逐个通关的前置课程。

### 当前维护状态也要分清

nanoGPT README 已加入 2025 年 11 月的说明：该仓库已老旧/弃用，建议看 nanochat。这不使旧教材失去学习价值，但意味着它不能被介绍成 Karpathy 当前唯一或最新的主线项目。[K3]

nanochat 的出现时间晚于你提到的 2024 年，因此它是本次基于目标补充的候选，而不是把你的历史记忆改写为 nanochat。[K5]

## 二、`build-nanogpt` 真正在教什么？

它从较少的代码开始，逐步搭建 GPT-2 模型，组织输入和目标，进行训练、评估和生成；随后涉及更有效率的 GPU 计算、梯度累积与多 GPU 训练。对有编程基础的学生，重要收获是把“模型训练”还原为可以逐行追踪的数据和程序，而不再只看到一个 API。[K1][K6]

完成相应练习后，你应能解释：

- 一段文字如何转成 token，怎样构造“前面的 token → 下一个 token”的训练目标。
- Transformer 的嵌入、因果注意力、前馈网络、残差连接如何组合。
- loss、梯度、优化器、批量、学习率和 checkpoint 分别负责什么。
- 为什么训练 loss 下降不等于在新任务上有效，为什么还要验证和基准评测。
- 为什么模型定义没有大改，改变计算方式和数据流仍可能显著影响运行效率。

**它主要是预训练教程，不是完整 ChatGPT 制作教程。**上游 README 明确提醒：训练出的模型是文本续写器，不是对话助手；聊天微调不在该视频范围内。没有 RL，也不能把载入预训练权重的演示当作从随机权重完成预训练。[K1]

### 对零 ML 基础的困难在哪里？

困难通常不是“你学不会”，而是教程会默认你已能看懂张量维度、反向传播、交叉熵及 PyTorch 基本写法。直接从 124M 和多 GPU 开始，容易同时遇到概念与工程两种阻碍。因此推荐先做 Zero to Hero 的必要部分和更小的 `ng-video-lecture`，再看这里。

## 三、为什么更推荐 `ng-video-lecture` 作为第一站？

它的 `gpt.py` 把字符编码、训练/验证切分、因果注意力、训练循环和采样放在一个短文件里，可以减小层数、宽度、序列长度与批量，先用 CPU 或消费级 GPU 观察 loss 和输出变化。与真实大模型相比，规模很小；与只调用模型 API 相比，你确实在更新自己模型的参数。[K2][K7]

原文件没有完整的 checkpoint 保存/恢复工作流，所以本仓库把“保存权重和字符表，再在不训练的情况下重新加载生成”列为补充练习。这样才能形成你自己的训练闭环，而不是只看一次终端输出。

## 四、MiMo 研究栈和 Karpathy 教程的核心对照

| 比较维度 | Karpathy 小 GPT / build-nanogpt | MiMo 9B + Agent/RL 资源 |
|---|---|---|
| 首要目标 | 让你理解并实现模型训练 | 让研究者研究已有模型的交互式后训练 |
| 起点 | 可以从随机初始化的参数开始 | 已经完成 SFT 的 Qwen 派生 9B |
| 一次训练样本 | 一段 token 和下一 token 目标 | 问题、环境、多步动作、工具反馈、奖励 |
| “学好”的主要信号 | 对正确下一 token 的预测损失 | 完成任务得到的奖励及样本外任务表现 |
| 主要代码关注点 | 张量、模型层、loss、backward、optimizer | 任务环境、采样、评分、调度和策略优化 |
| 最直接的学习收获 | 神经网络与语言模型训练的基本机制 | Agentic RL、奖励设计、评测与训练工程 |
| 是否讲解基础 | 有课程/视频逐步讲解；也有前置知识 | 面向已懂训练的研究者，基础讲解不足 |
| 是否适合第一台笔记本 | 缩小模型后适合；不是运行所有默认值 | 适合阅读与轻量轨迹实验，不默认适合完整 9B RL |
| 是否保证聊天能力 | 不保证；预训练项目主要续写 | 9B 已是 SFT 模型，但这是上游已训练的成果 |
| 能否代表完整工业流程 | 不能，但能做结构完整的小实验 | 不能仅凭代码发布就认定所有工业流程可完整复现 |

事实依据：[K1][K2][K6][M3][G1][G4][G5]；“适合谁”和推荐顺序是本仓库的学习设计判断。

**一句话：Karpathy 更适合回答“参数为什么会学”，MiMo 更适合回答“已经会做事的模型如何通过任务反馈继续学”。**

## 五、nanochat：针对“想走更完整流程”的补充选择

nanochat 比 build-nanogpt 更贴近“从训练到能聊天”的系统体验。仓库覆盖分词器、预训练、SFT、评测和聊天接口，还包含单独的 RL 脚本。[K5]

但有两个重要边界：

**第一，当前默认 speedrun 不包含 RL。**本次实际阅读 `runs/speedrun.sh` 和 `runs/runcpu.sh`，它们连接预训练、评测、SFT 等步骤；不能因为仓库有 `scripts/chat_rl.py`，就说默认一键脚本已经跑完强化学习。[K8][K9]

**第二，RL 脚本不是 MiMo 工业配方的缩小复刻。**`scripts/chat_rl.py` 使用 GSM8K 奖励；作者自己说明它相较常见 GRPO 做了简化，更接近 REINFORCE。它适合看“采样—评分—更新”的原理，不能把它和异步、多任务、长轨迹的 MiMo 训练视为同一实验。[K10]

你的更合适路线是：**小 GPT 入门 → nanochat 选择性体验预训练/SFT/聊天 → MiMo 任务与训练工程**。build-nanogpt 则放在想深入标准 GPT-2 架构和训练效率时，不设为强制中间关卡。

## 六、学习成本、时间与算力：必须分成三种时间

1. **视频时长：**只算播放时间，不包括暂停、推导、编程和排错。
2. **学习投入：**你实际看、写、改、解释和分析的时间。
3. **机器运行时间：**训练或评测程序占用设备的时间；它可以与学习重叠，但受配置与硬件影响。

Zero to Hero 官方列出 micrograd 2 小时 25 分、makemore 第一节 1 小时 57 分、《Let's build GPT》1 小时 56 分。这三段合计约 6 小时 18 分视频，不应写成“6 小时学完深度学习”。[K0]

| 项目与学习深度 | 主动学习时间：本仓库估算 | 算力定位与运行时间说明 |
|---|---:|---|
| 必要基础 + 小 GPT + 记录实验 | 24–37 小时 | CPU/消费级 GPU；每次先跑短实验再测算 |
| build-nanogpt 代码和缩小实验，已有基础后 | 12–20 小时 | 单卡可做裁剪实验；默认完整训练不是本地入门配方 |
| nanochat 小规模流程，已有基础后 | 6–12 小时 | CPU 路径只作教学，强能力 speedrun 使用多张高端 GPU |
| MiMo 模型卡和代码结构导读 | 已纳入基础路线 2–3 小时概览 | 不需要下载权重 |
| 一个 MiMo Agent 任务与评分链路，环境已可运行 | 4–8 小时 | 可调用远程模型，主要负担是容器、API 与排错；当前有配置缺口 |
| 自己设计并解释一个简单 RL 实验 | 8–16 小时，前提已具备训练基础 | 先使用能力足够的小模型和短任务，不能据此估算 9B 论文复现 |
| MiMo 9B 全量长程 RL 复现 | 不给虚假的固定完成时长 | 数据、依赖、模型支持、GPU 拓扑与评分器均须先验收 |

这些是学习计划的估算，不是实测承诺。基础较弱、依赖安装、网络下载和 GPU 排队会延长日历时间。

### 为什么不能把“作者只花了几美元”当成自己的预算？

build-nanogpt README 曾给出大约一小时/十美元的演示叙述，但本次读取的 `train_gpt2.py` 默认训练目标约为 **10B tokens、19,073 个优化步骤**；不同版本、数据量和硬件不能互换。先对齐脚本和训练目标，再谈时间。[K1][K6]

nanochat 当前 README 的参考运行约为 **8×H100、约两小时、约 48 美元**，而 speedrun 脚本注释写约 1.5 小时；这是作者特定配置的参考量级，不是今天的云 GPU 报价，也不是“用笔记本两小时学完”的承诺。CPU 教学脚本则降低目标能力，不是等价的性能复现。[K5][K8][K9]

实际预算与测算方法见 [算力与成本](05-resources-and-costs.md)。

[K0]: https://karpathy.ai/zero-to-hero.html
[K1]: https://github.com/karpathy/build-nanogpt
[K2]: https://github.com/karpathy/ng-video-lecture
[K3]: https://github.com/karpathy/nanoGPT
[K4]: https://github.com/karpathy/llm.c
[K5]: https://github.com/karpathy/nanochat
[K6]: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
[K7]: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
[K8]: https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh
[K9]: https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh
[K10]: https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py
[M3]: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
[G1]: https://github.com/XiaomiMiMo/mimoagent/tree/mimo-oss
[G4]: https://github.com/XiaomiMiMo/uni-agent/tree/mimo-oss
[G5]: https://github.com/XiaomiMiMo/uni-agent/blob/mimo-oss/examples/quickstart/training/train_qwen3p5_dense.sh
