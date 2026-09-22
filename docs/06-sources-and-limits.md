# 06｜来源索引、核验记录与未完成核验项

## 调研范围与时间

快照日期：**2026-09-22**，以官方发布文章的日期标识本次 MiMo V2.6 发布。不把媒体发布时间、GitHub 仓库创建时间、某个时区显示的“昨天”混为同一个事件时间。

目标是帮助尚未系统学习 ML/DL/AI Infra 的计算机专业学习者，辨认资料、比较教学项目、规划小规模动手实验。不是模型排行榜审计，也不是 MiMo 生产训练复现报告。

## 一、如何阅读证据等级

- **已读正文/代码：**本次实际读取了对应的官方页面、模型卡或指定源码。
- **已确认入口/文件：**确认资源链接或文件页面存在，但未成功取得、运行或逐页阅读全部内容。
- **官方宣布，交付物未完整核验：**官方文章有明确声明，但本次没有完成所有对应下载、镜像、数据及运行依赖的核验。
- **本仓库建议/估算：**教学顺序、学习时长、硬件分档和裁剪实验的设计，不是官方性能保证。

网页可读、源文件可读、程序能运行、分数能复现，是四个不同层级。

## 二、MiMo 一手来源

| 编号 | 官方来源 | 本次核验与用途 |
|---|---|---|
| M1 | [中文发布说明](https://mimo.mi.com/docs/zh-CN/news/latest/v2-6) | 已读。核验发布日期、资源范围、7k+ 环境声明、9B 后续 RL 实验介绍 |
| M2 | [英文发布文章](https://mimo.xiaomi.com/mimo-v2-6/article) | 已读。核验 Live RL、正式模型发布及报告入口 |
| M3 | [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) | 已读模型卡。确认 Qwen3.5-9B、MiMo 数据监督微调、公开检查点为 SFT |
| M4 | [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | 已读模型卡。架构口径、许可证、方法摘要与部署入口 |
| M5 | [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 已读模型卡。参数与部署口径，不将激活参数当显存预算 |
| M6 | [技术报告 PDF 文件页面](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf) | 已确认文件与官方链接；PDF 下载/渲染未成功，未完成全文精读 |
| M7 | [Live RL 仪表盘](https://mimo.xiaomi.com/rl/) | 已确认入口，当前获取到的内容不足以独立核验交互曲线；六天运行依据官方文章 |
| M8 | [MiMo V2.6 Collection](https://huggingface.co/collections/XiaomiMiMo/mimo-v26) | 官方资源总入口；正文分别链接具体模型，而非只给集合页 |
| M9 | [XiaomiMiMo Hugging Face 组织](https://huggingface.co/XiaomiMiMo) | 用于资源交叉检查；组织列表不等于对所有数据交付物的穷尽证明 |

### 关于罗福莉的 X 原帖

账号入口：[Fuli Luo / @_LuoFuli](https://x.com/_LuoFuli)。检索定位到的相关原帖候选：[status/2100296686719610932](https://x.com/_LuoFuli/status/2100296686719610932)。

**本次直接访问受限，未直接核验该原帖完整内容。**因此不逐字引用，不声称读完完整线程，不对每日训练事件做未经证实的还原。Live RL 的存在与正式发布事实使用小米官方发布文章交叉确认。转载仅用来定位入口，不承担本文技术结论的证据。

## 三、小米 GitHub 源码

| 编号 | 仓库/文件 | 本次核验与用途 |
|---|---|---|
| G1 | [mimoagent / mimo-oss](https://github.com/XiaomiMiMo/mimoagent/tree/mimo-oss) | 已读 README；区分 Agent rollout 与参数更新 |
| G2 | [mimoagent example_configs 目录](https://api.github.com/repos/XiaomiMiMo/mimoagent/contents/example_configs?ref=mimo-oss) | 已检查目录及精确文件访问；README 所引 swe_docker.yaml 读取返回 404 |
| G3 | [mini-swe-agent.yaml](https://github.com/XiaomiMiMo/mimoagent/blob/mimo-oss/example_configs/mini-swe-agent.yaml) | 已读；当前为 Kubernetes 后端，cost_limit=0.0 表示不限额，不当作 Docker 替代 |
| G4 | [uni-agent / mimo-oss](https://github.com/XiaomiMiMo/uni-agent/tree/mimo-oss) | 已读 README 并检查 quickstart 目录；识别 gateway、训练和推理示例 |
| G5 | [train_qwen3p5_dense.sh](https://github.com/XiaomiMiMo/uni-agent/blob/mimo-oss/examples/quickstart/training/train_qwen3p5_dense.sh) | 已读相关代码；确认默认 Qwen3.5-4B、8×8 GPU、128K 响应和并发参数 |
| G6 | [verl / mimo-oss](https://github.com/XiaomiMiMo/verl/tree/mimo-oss) | 已读仓库说明；作为 RL 优化组件定位，不声称审计全部修改 |

mimoagent 读取到的分支提交：`467f0a19016f0ac4d63b8d17a1f0da9ba07f232c`。

uni-agent 训练脚本读取到的 blob SHA：`046501e144df62fe4337da7c12c59ea8d47b0d05`。**这是文件内容标识，不是可以当成仓库版本执行 `git checkout` 的 commit SHA。**可由 [Git blob API](https://api.github.com/repos/XiaomiMiMo/uni-agent/git/blobs/046501e144df62fe4337da7c12c59ea8d47b0d05)核对该文件快照。

分支和 README 可能继续更新；自己实际开跑时，再记录当前 `git rev-parse HEAD` 和依赖锁文件。不要将上游 README 中其他模型、其他项目的结果自动归为 MiMo 9B 结果。

## 四、Karpathy 与 PyTorch 一手来源

| 编号 | 来源 | 主要支持的内容 |
|---|---|---|
| K0 | [Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 课程前提、基础教学顺序、官方视频时长 |
| K1 | [build-nanogpt](https://github.com/karpathy/build-nanogpt) | GPT-2 复现视频定位，非聊天微调教程 |
| K2 | [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | 小 GPT 视频配套仓库 |
| K3 | [nanoGPT](https://github.com/karpathy/nanoGPT) | 训练器定位、2025 年 11 月的弃用/迁移说明 |
| K4 | [llm.c](https://github.com/karpathy/llm.c) | C/CUDA 训练与底层效率方向 |
| K5 | [nanochat](https://github.com/karpathy/nanochat) | 聊天模型端到端项目、作者参考算力与成本 |
| K6 | [build-nanogpt/train_gpt2.py](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py) | 训练目标、批量、步数、梯度累积与 checkpoint 范围 |
| K7 | [ng-video-lecture/gpt.py](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) | 字符编码、模型、训练/验证和采样；用于制定缩小实验 |
| K8 | [nanochat/runs/speedrun.sh](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh) | 多 GPU 参考流程；当前脚本不包含 RL 阶段 |
| K9 | [nanochat/runs/runcpu.sh](https://github.com/karpathy/nanochat/blob/master/runs/runcpu.sh) | CPU 教学配置与流程，不等价于强能力 speedrun |
| K10 | [nanochat/scripts/chat_rl.py](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py) | GSM8K、加载 SFT checkpoint、简化 RL、采样参数关系 |
| P1 | [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) | 按操作系统和硬件选择兼容安装，而非照抄旧版命令 |
| P2 | [PyTorch 模型保存与加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | state_dict、优化器状态、推理/恢复的区别 |

本仓库没有完整镜像这些项目、下载其全部数据或执行它们的默认训练。用于教程修改的代码片段是本次学习方案的一部分，不声称已在目标硬件完成运行验证。

## 五、仍未完成核验的关键事项

### 1. 技术报告全文

确认官方 PDF 文件存在，但完整字节下载与页面读取失败。没有使用二手“报告解读”填补成貌似读过全文的结论。正文提供的是基于官方发布文章和模型卡的导读，而不是逐章精读报告。

### 2. 7k+ 任务的完整交付清单

官方已经作出开放声明，但本次未完成全部数据路径、任务镜像、许可证、评分器与训练配方之间的一一对应。不能下结论说“从未发布”，也不能声称“所有资源已经下载验收”。

### 3. mimoagent Docker quickstart

精确配置路径读取返回 404，说明当前不能把该 README 步骤称为已核验的可运行第一课。其余成功读取的仓库资源不受此影响。文件可能随着后续发布补齐，复查时应记录日期和提交。

### 4. 真正的训练与性能

没有在用户设备训练，没有运行 9B RL，没有做独立 benchmark。厂商公开的提升数字标为官方报告，不当作本次实测。学习时长、资源分档与费用公式都是附前提的规划/推导。

## 六、GitHub 交付记录

开始全面调研前，先验证了目标仓库的访问与写入权限。目标是新建空仓库，初次内容读取的“仓库为空”不是授权失败。初始化写入成功后才继续资料整理。

初始化提交：[`9aa3a26b8f375408cc5c2c4dc67cae173328a8a0`](https://github.com/LilJay-H/mimo-v2.6-tech-learning/commit/9aa3a26b8f375408cc5c2c4dc67cae173328a8a0)。后续研究文档与导航在同一仓库的 main 分支分批提交，完整记录见 [提交历史](https://github.com/LilJay-H/mimo-v2.6-tech-learning/commits/main)。

本次仅添加学习资料与模板，没有修改小米或 Karpathy 的上游仓库，没有启动任何付费训练或自动化监测。
