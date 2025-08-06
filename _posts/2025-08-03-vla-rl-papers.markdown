---
layout: post
title:  "看看论文"
date:   2025-08-03 00:14:00 +0800
categories: posts
tag: autonomous driving
---

# B2DVL完事了，看看最近的论文来搞新东西

## Diffusion

### Reinforcement Learning for Flow-Matching Policies  
Samuel Pfrommer, Yixiao Huang, Somayeh Sojoudi（UC Berkeley）  
arXiv:2507.15073v1 [cs.LG] 20 Jul 2025  [Reinforcement Learning for Flow-Matching Policies](https://arxiv.org/abs/2507.15073)

这个论文到现在两个月了，github repo 仅有 4 star，难说靠不靠谱。看起来就是给 flow matching 上了 GRPO 来试试效果。

本文首次系统地把强化学习（RL）用于“流匹配（flow-matching）”策略的二次优化，使其能够超越次优演示数据，并支持可变时域的最短时间控制；提出了两种方法——RWFM（加权模仿）与 GRPO（组相对策略优化），在仿真独轮车任务上将演示策略成本降低 50–85%。

1. 研究背景  
• 现代通才机器人策略（VLA 模型，如 π0、RT-2、Octo）普遍采用“流匹配 / 扩散”动作专家，通过模仿学习一次性生成固定长度动作块。  
• 但人类演示存在两类次优：  
    – 变异性次优（variation suboptimality）：同一指令下动作优劣不一。  
    – 支撑集次优（support suboptimality）：演示未覆盖更优动作轨迹。  
• 固定长度动作块也无法实现“尽快完成任务”的最短时间目标。
2. 主要贡献  
(1) 形式化流匹配策略的 RL 问题，指出需同时解决上述两类次优。  
(2) 在流匹配框架中引入“时域通道”，让模型可生成任意长度的动作轨迹，实现可变时域规划。  
(3) 提出两种 RL 微调方法：  
    • RWFM（Reward-Weighted Flow Matching）：在模仿损失中用指数奖励加权，并加入高斯“bump”探索噪声，使策略逐渐偏向高奖励轨迹并突破演示支撑集。  
    • GRPO（Group Relative Policy Optimization）：  
    – 采用免价值函数的 PPO 变体 GRPO，提升样本效率；  
    – 训练一个轻量级奖励代理 Rφ( ̃o, A)，在无需真实 rollout 的情况下即可估计动作块奖励；  
    – 组内相对优势计算 + 加权流匹配损失，显著减少与环境交互次数。  
(4) 在 6 种奖励设定（位置、时间、速度、碰撞、朝向、控制正则）的 2D 独轮车环境上验证：  
    • 纯模仿（ILFM）只能复现演示水平；  
    • RWFM 显著提升，但仍受限于在线 rollout 成本；  
    • GRPO 用更少交互量达到最佳，平均成本比 ILFM 低 50–85%。  
(5) 消融实验显示：  
    • 奖励缩放 α 过大易导致策略忽视观测条件；  
    • 动作探索幅度 M≈0.2 对突破支撑集最有效，能让策略学会演示中未出现的“刹车”行为。
3. 方法要点  
• 可变时域：把原始轨迹插值到统一长度 H′，并拼接“期望时域”额外通道；生成时再反推真实步数。  
• RWFM：在演示数据上按 e^{αR} 加权做流匹配，随后用当前策略收集高奖励轨迹继续训练。  
• GRPO：  
    – 预训练阶段先用演示数据训练奖励代理 Rφ（TimesNet 架构）。  
    – 每轮采样 G=10 条动作块，用 Rφ 计算组内优势，加权更新流匹配网络。  
    – 仅在验证性能停滞时才进行真实环境的少量 rollout，以校正奖励代理。



### Transition Matching: Scalable and Flexible Generative Modeling

Meta

Flow Matching 的改进，[airxiv](https://arxiv.org/abs/2506.23589)

这篇文章的话，已经有珠玉在前。有一篇讲 transition matching 的知乎文章，顺带讲了一些 flow matching 的原理： [Transition Matching: Scalable and Flexible Generative Modeling （公式看晕了，喂！） - 知乎](https://zhuanlan.zhihu.com/p/1928847692173390104)

另外记一下 velocity：

在 **Flow Matching** 中，**velocity（速度场）** 是一个向量场，描述了如何从噪声（源分布）一步步“流动”到真实数据（目标分布）。

在 Flow Matching 中，我们假设：
- 有一个**源分布** $p_0(x)$（通常是标准高斯）；
- 有一个**目标分布** $p_1(x)$（真实数据分布）；
- 我们希望构造一条**连续的路径**（flow），把 $p_0$ 变成 $p_1$。

这个路径是通过一个**常微分方程（ODE）**来定义的：

$$
\frac{dx_t}{dt} = v(x_t, t), \quad t \in [0, 1]
$$

其中：
- $x_t$ 是时间 $t$ 时的中间状态；
- $v(x_t, t)$ 就是所谓的 **velocity field（速度场）**，它告诉我们在每个位置 $x_t$ 和时间 $t$ 应该朝哪个方向“移动”。

假设现在有一个点 $x_0$ 从标准高斯采样出来，目标是变成一张猫的图片 $x_1$。

- **velocity $v(x_t, t)$** 就是每一步告诉你：
  
  > “你现在在这个位置 $x_t$，下一步应该往哪个方向走，才能更接近真实的猫图？”

这个速度场是用一个神经网络 $v_\theta(x_t, t)$ 来学习的。

| 方法                      | 如何移动                                      |
| ------------------------- | --------------------------------------------- |
| **扩散模型（Diffusion）** | 通过加噪/去噪，每一步是随机扰动（SDE）        |
| **Flow Matching**         | 通过确定性速度场，每一步是**直接移动**（ODE） |



### Steering Your Diffusion Policy with Latent Space Reinforcement Learning

Wagenmaker 等，UC Berkeley & UW

arXiv:2506.15799v2 [cs.RO] 25 Jun 2025  [arxiv](https://arxiv.org/abs/2506.15799v2)  [github](https://diffusion-steering.github.io/) 

一句话总结
DSRL 把 RL 的作用域从“改权重”变成了“改噪声”：在保持预训练扩散/流匹配策略权重冻结的前提下，仅在其输入噪声空间里学一个小策略，就能在极少交互下把成功率从 20 % 提到 90 %，并首次实现了 π0 这类 3.3 B 通用机器人策略的在线 RL 微调。

1. 研究动机
   • 行为克隆（BC）+ 扩散策略（πdp）已成为通才机器人控制的主流，但遇到新场景往往表现不佳。
   • 传统 RL 微调需更新大模型权重，代价高且不稳定；后处理动作或残差策略又效率低。
   • 关键观察：扩散模型是“给定初始噪声→确定性去噪→动作”的映射，因此只需控制噪声即可控制动作。
2. 核心思路：Diffusion Steering via RL (DSRL)
   • 把原 MDP 重写成“噪声-动作”MDP M^w：状态 s → 选噪声 w → πdp(s,w) 输出动作 a。
   • 在 M^w 上用任意 RL 算法学一个小策略 π_w(s) → w，完全不碰 πdp 权重。
   • 好处：
   – 黑盒、无需反向传播多步去噪链；
   – 参数量缩小几个数量级，可在真实机器人 1 GPU 上在线训练；
   – 即插即用，适用于扩散/流匹配、DDIM/DDPM、专有或开源大模型。
3. 方法细节
   (1) 噪声别名 (Noise Aliasing) 技巧
   – 同一动作可能由不同 w 得到，利用这一点把离线数据转成 w-Q 值，提升样本效率；
   – 提出 DSRL-NA：两个 critic（Q_A 在动作空间，Q_W 在噪声空间），Q_W 通过 Q_A 蒸馏 + 别名映射，实现离线/在线统一。

(2) 训练流程
– 预训练 πdp（冻结）→ 在噪声空间初始化 π_w、Q_w → 在线/离线 RL 更新 π_w。
– 仅需前向推理 πdp，无需梯度回传。

1. 实验亮点
   • 在线适应：在 Robomimic、OpenAI Gym、真实 Franka / WidowX 上，DSRL 用 5–10× 更少交互即可超过 DPPO、IDQL 等最新方法。
   • 离线适应：在 10 个 OGBench 任务中，DSRL 有一半任务是 SoTA，且只需用同一份离线数据。
   • 真实世界：
   – 单任务 Cube→Bowl：BC 20 % → DSRL 90 %（<50 回合）。
   – 多任务 Bridge-V2 预训练：DSRL 在 100–150 回合内显著改善抽屉、堆叠等任务。
   – π0（3.3 B）微调：Libero 任务 20 %→100 %（∼1 万步），Aloha 双手机器人任务显著超越 RESIP/V-GPS；首次在真实 Franka 上成功微调 π0。
2. 消融与洞察
   • 噪声别名可把在线样本效率再提高 2×。
   • 即使 πdp 训练数据质量差、过拟合或网络小，DSRL 仍能快速“拉回”性能。
   • 去噪步数、网络规模对 DSRL 几乎无影响；限制噪声幅值 b_w∈[1,3] 即可保证稳定性。
3. 局限与未来方向
   • 若 πdp 动作分布极度集中，可能无法提供足够可操纵的噪声空间；
   • 仍需奖励设计与环境重置；
   • 未来可探索：
   – 将“噪声”拓展到 prompt / 观测空间；
   – 理论刻画噪声空间表达能力；
   – 无奖励或自动奖励设定。

DSRL 把“微调大模型”变成“微调小噪声”，让真实机器人用几十次交互就能把 20 % 成功率飙到 90 %，并首次把 RL 成功塞进 3.3 B 参数的通用策略 π0，为现场自适应打开了实用大门。



### Hierarchical Rectified Flow Matching with Mini-Batch Couplings

[github repo](https://github.com/riccizz/HRF_coupling) 还没开源而且仅有一个 star。

Yichi Zhang, Yici Yan, Alex Schwing, Zhizhen Zhao  
University of Illinois Urbana-Champaign  

提出了一种**改进的流匹配方法（HRF）**，通过**小批量最优传输（mini-batch optimal transport）**来**简化速度场的多模态分布**，从而提升生成质量和效率，尤其适用于**低计算预算（低NFE）场景**。

**流匹配（Flow Matching）** 是一种生成模型方法，通过建模一个**速度场（velocity field）** 来将简单分布（如高斯）转换成复杂数据分布。它通过数值积分一个常微分方程（ODE）来生成样本。

但传统流匹配存在两个问题：
1. **速度场是多模态的**，难以建模；
2. **路径弯曲**，导致采样效率低。

**层次流匹配（Hierarchical Rectified Flow, HRF）** 被提出用于建模速度场的分布，但它在每一层的复杂度都很高，**没有简化建模过程**。

作者提出：**通过小批量耦合（mini-batch couplings）来逐步简化速度场的复杂度**，从而改善生成质量和效率。具体包括：

**数据空间耦合（Data Coupling）**
- 不再独立采样源分布和数据点，而是**用小批量最优传输（mini-batch OT）来配对样本**。
- 结果：速度场分布变得更**单峰（unimodal）**，更容易学习。

**速度空间耦合（Velocity Coupling）**
- 在速度空间中，也用小批量OT来配对初始速度 $v_0$ 和目标速度 $v_1$。
- 结果：采样路径更直，**减少积分步数（NFE）**，尤其适用于**低预算采样**。

**联合耦合（Data + Velocity Coupling）**
- 两阶段训练：先用数据耦合训练，再用速度耦合微调。
- 结果：在低NFE（如1步）下也能生成高质量样本。

之后有需要了再细看吧。。



## RL4AD

### CaRL: Learning Scalable Planning Policies with Simple Rewards

Bernhard Jaeger et al. University of Tübingen）
熟悉的作者，熟悉的配方

[arxiv](https://arxiv.org/pdf/2504.17838) [github](https://github.com/autonomousvision/CaRL)

有一说一，这个工作非常 straightforward。RL是非常 consume rollout 次数的，我觉得 CARLA 直接练还是很难加速，资源消耗太大了，必须尽量避免如此直接的方法，除非有什么魔法能不用练几下就能达到很好的效果，比如有一个很好的 backbone VLA 只需要少许微调，不用在 CARLA 上从零开始。但是他这个说是 Due to these optimizations, we were able to reproduce Roach with 10 million environment samples in 32 hours on a single A100 GPU. 如果这样的话，倒还好。

> **CaRL 提出了一种极简的强化学习奖励设计，仅用“路线完成度”+惩罚项，就能在大规模数据下训练出高效、鲁棒的自动驾驶规划策略，在 CARLA 和 nuPlan 上都取得了 SOTA 性能。**

#### 极简奖励设计（Simple Reward）

> **只用一个奖励项：路线完成度（Route Completion, RC）**

#### 奖励公式：

  ```
  reward = RC * ∏(soft penalty) - terminal penalty
  ```


- **软惩罚**：如超速、偏离车道、舒适性等，乘性衰减；

- **硬惩罚**：如碰撞、闯红灯，直接终止 episode；

- **无规则参考**：不依赖任何规则系统，避免性能上限。

#### 可扩展训练（Scalable Training）

- **PPO 在大 batch size 下失效**：复杂奖励在大 batch 下容易陷入局部最优；
- **简单奖励可扩展**：在大 batch 下性能反而提升；
- **训练规模**：
  - CARLA：300M samples（30× 以往工作）
  - nuPlan：500M samples
  - 使用 8-GPU 单机训练，DD-PPO 分布式训练框架

#### 工程优化

- **CARLA 训练加速**：
  - 不重启 town、预计算 A* 路径
  - 异步数据收集（AC-PPO）
  - 场景自动生成脚本
- **nuPlan 适配**：
  - 增加 survival bonus 防止提前完成任务后“摆烂”



### Action Space Reduction Strategies for Reinforcement Learning in Autonomous Driving

[arxiv](https://arxiv.org/abs/2507.05251) 暂未开源...

是一个在 CARLA 里 RL 的小改进，就是减小动作空间。具体在训练上也是直接 PPO，倒是创新性有限（？）。

> **论文提出两种新方法（动态掩码 + 相对动作）来“聪明地砍掉无用动作”，**  
> **在 CARLA 仿真中让 PPO 智能体训练提速 2 倍，成功率更高，驾驶更平滑。**

#### 提出的两种新方法

| 方法                   | 核心思想                                                     | 动作空间大小                         | 特点                                             |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------ | ------------------------------------------------ |
| **Dynamic Masking**    | **每帧只保留当前方向盘 ±0.2 范围内的 5 个转向 + 2 个油门**，其余动作被二进制掩码设为无效 | 仍为 42 维，但有效动作只剩 5×2=10 个 | 实时、上下文相关，避免无效探索                   |
| **Relative Reduction** | **动作改为“相对调整”**：{-0.2, -0.1, 0, 0.1, 0.2}，加到当前方向盘上** | 固定 5×2=10 维                       | 动作空间维度不变，但物理意义更平滑，天然限制越界 |

#### 实验验证（CARLA Town07）

| 指标           | 结果                                                     |
| -------------- | -------------------------------------------------------- |
| **训练时间**   | 动态/相对方法比“全动作空间”**快 2 倍**                   |
| **成功率**     | **Rel-0.5** 和 **Dyn-0.5** 在大多数场景 ≥ 70%，甚至 100% |
| **控制平滑性** | 车道偏离显著降低（Rel-0.5 仅 0.10 m vs 全动作 0.31 m）   |
| **复杂场景**   | 多转弯、十字路口也能稳定完成                             |


#### 关键结论

- **动作空间不是越大越好**——合理缩减反而提升性能。
- **动态掩码** 和 **相对动作** 在 **速度、成功率、平滑性** 之间取得最佳平衡。
- **可移植性高**：方法独立于网络结构，可插拔到任何 RL 框架（PPO、SAC、DDPG 等）。





## VLA

### Pre-training Auto-regressive Robotic Models with 4D Representations

**ARM4R** ——用“人视频里学 4D 轨迹（3D 点+时间）”来预训练机器人策略，结果只用 1/10 的机器人数据就超过了 OpenVLA、π0-FAST 等 SOTA 方法说是。而且他这个用的资源也不多："Finally, we use 4 NVIDIA A6000 GPUs for training and a single NVIDIA A6000 GPU for evaluation."

先插一嘴，它的网站做得不错，分两栏，左边图片右边文字。虽然也是套模板，但是做了点小改动：[website](https://arm4r.github.io/) 但是看多了也就那样，对手机用户不太友好。

ARM4R 的完整模型 =「四路编码器 + 一个自回归 Transformer + 解码器」。下面给出可直接落地的**架构细节**与**训练脚本骨架**（PyTorch 伪码），按三阶段顺序展开。  

──────────────────  

#### 模型架构

```text
输入 (t 时刻)
├─ 语言指令 l               → 冻结 CLIP-T 文本编码器 → z_l
├─ 图像 i_t                 → 冻结 ViT-B/16           → z_i
├─ 3D 点/机器人状态 p_t     → 2 层 MLP                → z_p
└─ 历史信息                → 拼接后喂给 Causal Transformer

Transformer：随机初始化 ViT-Base（12 层、768 维、8 头），因果注意力，窗口 C=16（或32）。  

输出
├─ 未来 3D 点 p_{t+1}（Stage1/2）或未来机器人状态 s_{t+1}（Stage3）
└─ 解码：2 层 MLP 直接回归，L1 loss
```

####  数据与预处理  

| 阶段   | 数据                                     | 伪标签生成                                | 采样率 |
| ------ | ---------------------------------------- | ----------------------------------------- | ------ |
| Stage1 | Epic-Kitchens100 76k 视频                | SpatialTracker 产生 3D 点轨迹（g×g 网格） | 10 fps |
| Stage2 | 1–2 k 机器人演示视频                     | 同上                                      | 10 fps |
| Stage3 | 190 × 任务变体 成功轨迹（末端位姿+夹爪） | 无伪标签，直接用机器人真值                | 10 fps |

把整个流程封装成 **「人视频→3D 轨迹预训练」→「机器人视频→3D 轨迹微调」→「机器人动作→控制微调」**，只需替换最后一步的 MLP 头即可。

#### 训练过程

──────────────────  
阶段 1：人视频 4D 轨迹预训练  
• 数据  
  – 76 k 条 Epic-Kitchens100 egocentric 人视频（75 041 条有效）。  
  – 无人工动作标签，只用 **伪 3D 点轨迹**：  
    - 在首帧布 g×g 网格 → SpatialTracker 产生每帧 3D 坐标（相机坐标系）。  
• 训练任务  
        – 「给定语言指令 + 当前图像 + 当前 3D 点 p_t → 预测 t+1 的 3D 点 p_t+1」。  
        – 采用 **自回归 next-token 范式**，损失为 L1(p̂_t+1, p_t+1)。  

──────────────────  
阶段 2：机器人场景 4D 轨迹微调（一次即可，跨任务共享）  
• 数据  
  – 每条任务仅 **5–10 % 阶段 1 数据量**，即 1–2 k 段机器人演示视频。  
  – 仍用 SpatialTracker 产生 3D 点轨迹，但相机固定，场景为机器人。  
• 训练任务  
  – 与阶段 1 相同：预测 3D 点 → 解决人→机器人相机/场景分布差异。  

──────────────────  
阶段 3：机器人控制微调（任务专用）  
• 数据  
  – **190 段成功演示/任务**（比基线少 10×）。  
  – 观测：语言指令 + 图像 + **机器人当前状态 s_t（末端位姿+夹爪）**。  
  – 标签：下一时刻状态 s_t+1。  
• 训练任务  
  – 把阶段 1/2 的 **3D 点输入/输出通道** 换成 **机器人状态通道**；  
  – 仍用自回归 Transformer，损失 L1(ŝ_t+1, s_t+1)。  
  – 预测 16 步，执行第 1 步（滚动时域）。  

──────────────────  
把「人视频里学 3D 点轨迹」→「机器人视频里继续学 3D 点轨迹」→「把预测目标换成机器人状态」，总共使用的数据有：

1. 大量无标人视频（+ SpatialTracker 伪轨迹）；  
2. 少量机器人演示视频（同伪轨迹）；  
3. 目标任务少量成功轨迹（用于阶段 3）。



### MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation

[arxiv](https://arxiv.org/abs/2507.10543) 

北大的工作，刚开源三周左右，10 stars


> **MP1 是第一个将 MeanFlow（平均速度场）引入机器人操作任务的方法，**  
> **仅用 1 步推理（1-NFE）即可输出高质量动作轨迹，**  
> **在多个仿真与真实任务中显著优于扩散模型（如 DP3）和流模型（如 FlowPolicy）。**


| 方法类型 | 优点 | 缺点 |
|---|---|---|
| **Diffusion Policy（如 DP3）** | 多模态、鲁棒性强 | 推理慢（需 10+ NFE） |
| **Flow Policy（如 FlowPolicy）** | 1-NFE 推理快 | 需一致性约束，误差大 |
| **MeanFlow（图像生成）** | 1-NFE、无需一致性 | 尚未用于机器人任务 |

> ❗ **MP1 首次将 MeanFlow 引入机器人学习，解决“快 vs 准”的权衡问题。**


| 模块 | 设计 | 作用 |
|---|---|---|
| **MeanFlow Identity** | 直接建模平均速度场，绕过 ODE 积分 | 实现真正 1-NFE 推理 |
| **Dispersive Loss** | 无正样本的对比正则项，拉开不同状态嵌入 | 提升少样本泛化能力 |
| **CFG（Classifier-Free Guidance）** | 增强可控性 | 不影响 1-NFE 推理速度 |
| **3D Point Cloud Input** | 使用 512/1024 点云作为视觉输入 | 提升空间理解能力 |

#### 原理

**输入处理**

- **视觉输入**：3D 点云 → 3D Projection → 特征向量 `fv`
- **状态输入**：机器人关节状态 → 全连接层 → 特征向量 `fs`
- **条件向量**：`c = concat(fv, fs)`

**MeanFlow 建模**
```text
u(z_t, r, t) = (1/(t-r)) ∫_r^t v(z_τ, τ) dτ
```
- 通过 **MeanFlow Identity** 直接回归平均速度场，无需 ODE 求解；
- 训练目标为：
```
L_cfg = ||uθ - sg(u_target)||²
```

**Dispersive Loss（正则项）**
```text
L_disp = log E[exp(-||zi - zj||² / τ)]
```
- 无正样本的对比损失；
- 训练时增强表征区分度，推理时不增加成本。

**最终损失**
```
L_total = L_cfg + λ·L_disp
```

#### 实验结果

**仿真任务（Adroit + Meta-World，共 37 个任务）**

| 方法 | NFE | 平均成功率 | 平均推理时间 |
|---|---|---|---|
| **DP3** | 10 | 68.7% | 132.2 ms |
| **FlowPolicy** | 1 | 71.6% | 12.6 ms |
| **MP1 (ours)** | 1 | **78.9%** | **6.8 ms** |

> ✅ **MP1 比 DP3 快 19×，比 FlowPolicy 快 2×，且成功率提升 7.3%。**

**真实世界任务（ARX R5 双臂机器人）**

| 任务 | MP1 成功率 | 完成时间 |
|---|---|---|
| Hammer | 90% | 18.6s |
| Drawer Close | 100% | 8.8s |
| Heat Water | 90% | 23.4s |
| Stack Block | 80% | 27.2s |
| Spoon | 90% | 22.6s |

> ✅ **MP1 在所有任务中成功率最高，完成时间最短。**


#### 数据需求

| 数据项 | 格式 | 说明 |
|---|---|---|
| **专家轨迹** | 每帧一条记录，包含：<br>• 点云 `P ∈ ℝ^(N×3)`<br>• 机器人状态 `S ∈ ℝ^s`（关节角/末端位姿）<br>• 动作序列 `A ∈ ℝ^(K×a)` | `N=512/1024` 点，`K=4` 步预测，`a` 为动作维度 |
| **训练集规模** | 10 条演示即可收敛，20 条以上收益递减 | 与 DP3 / FlowPolicy 设置一致 |
| **数据来源** | 仿真（Adroit、Meta-World）或真实机器人（ARX R5） | 仿真用 Isaac Gym / MuJoCo，真实用 ROS 采集 |

#### 资源需求

All training and testing are performed on an NVIDIA RTX4090 GPU,with a batch size of 128, optimization uses the AdamW optimizer with a learningrate of 0.0001 (Adroit and Meta-World apply the same learning  rate), an observation window of 2steps, a history length of 4 states, and a prediction horizon of 4 steps.



### DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge

[website 标题跳动动画挺萌的](https://zhangwenyao1.github.io/DreamVLA/) [github刚刚开源一个月已经139stars](https://github.com/Zhangwenyao1/DreamVLA) [arxiv](https://arxiv.org/abs/2507.04447)

> 典中典之先思考，用了 \<dream\>思考过程\<action\> 不过如果迁移到自动驾驶，感觉不太适用。练这个需要的数据模态太多，耗费资源大，再加上我觉得它更擅长于处理静态的场景...

这篇论文《DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge》提出了一种**新型的机器人操作策略框架**，旨在通过**预测未来世界知识**（如动态区域、深度图、语义信息）来增强机器人的**感知-预测-行动闭环能力**，从而提升其在复杂任务中的泛化性和推理能力。

#### 模型组件

| 模块                            | 作用                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| **World Knowledge Forecasting** | 预测未来关键信息：动态区域（哪里会动）、深度（3D结构）、语义（物体类别） |
| **Block-wise Attention**        | 防止不同信息（动态、深度、语义）互相干扰，保持表示干净       |
| **Diffusion-based Action Head** | 用扩散模型从“世界知识”中解码出连续动作序列                   |
| **端到端训练**                  | 统一训练视觉、语言、预测和动作，无需额外生成模型             |

#### 模型改进

| 传统方法                             | DreamVLA 的改进                        |
| ------------------------------------ | -------------------------------------- |
| 直接从图像+语言 → 动作，缺乏未来推理 | 先预测“未来世界状态”，再决定动作       |
| 预测整帧图像，冗余信息多             | 只预测关键区域（动态区域、深度、语义） |
| 多模态信息混杂，互相干扰             | 用结构化注意力隔离不同信息             |
| 动作预测不稳定                       | 用扩散模型生成平滑、物理合理的动作序列 |

#### 实验结果

| 场景                        | 结果                                                        |
| --------------------------- | ----------------------------------------------------------- |
| **仿真 CALVIN 基准**        | 平均任务长度 **4.44**（SOTA），优于 OpenVLA、GR-1 等方法    |
| **真实机器人（Franka 臂）** | 成功率 **76.7%**，显著高于 Diffusion Policy、OpenVLA 等基线 |

#### 所需数据

**📦 数据类型（多模态）**

| 模态 | 内容 | 格式 |
|------|------|------|
| **视觉** | RGB 图像（静态摄像头 + 手腕摄像头） | JPEG/PNG |
| **深度** | 单目深度图（可选） | PNG 或 NumPy 数组 |
| **语言** | 自然语言任务指令 | JSON 或纯文本 |
| **动作** | 连续 7 维动作（6D 位移 + 夹爪状态） | NumPy 数组 |
| **状态** | 机器人末端位姿 + 夹爪状态 | NumPy 数组 |
| **辅助标签** | 动态区域掩码、SAM 分割、DINOv2 特征 | NumPy 数组 |


**数据来源与量级**

| 阶段 | 数据集 | 数据量 | 说明 |
|------|--------|--------|------|
| **预训练** | **DROID** | **7.6 万条机器人轨迹** | 多样化真实场景，Franka 机器人操作 |
| **预训练** | **CALVIN（无语言子集）** | 约 240 万步交互 | 仿真环境，长周期任务 |
| **微调** | **任务特定演示** | **每任务约 100 条轨迹** | 真实机器人任务，如拾取、放置、抽屉开关 |

#### 训练流程

**阶段 1：预训练**
- **目标**：让模型学会从多模态输入中预测“未来世界知识”（动态区域、深度、语义）。
- **数据**：DROID + CALVIN（无语言）。
- **监督信号**：
  - 动态区域：用 CoTracker 提取光流掩码
  - 深度：Depth-Anything 伪标签
  - 语义：DINOv2 + SAM 特征
- **训练设置**：
  - 优化器：AdamW，学习率 1e-3，余弦调度，5% 预热
  - 批量大小：8（每 GPU）× 8 GPU = 64
  - 训练轮数：20 epochs
  - 损失权重：
    - 动态区域：λ_dyn = 0.1
    - 深度：λ_depth = 0.001
    - 语义：λ_sem = 0.1
    - 动作：λ_DiT = 1

**阶段 2：微调**
- **目标**：适应具体任务（如拾取、放置、抽屉操作）。
- **数据**：每个任务约 100 条真实机器人演示。
- **训练方式**：
  - 继续使用预训练权重
  - 只训练解码器和动作头，冻结部分主干（可选）
  - 选择验证成功率最高的 checkpoint 作为最终模型


> **训练 DreamVLA 需用 DROID（7.6万条）+ CALVIN 预训练，再用每任务约 100 条真实演示微调，数据需包含 RGB、语言、动作、动态掩码、深度和语义特征，训练采用两阶段策略，耗时约 20 epochs，8×A800 GPU。**




## 场景重建

四维重建要在三维基础上再加时间，有点猛了

### StreamVGGT: Streaming 4D Visual Geometry Transformer

Dong Zhuo*, Wenzhao Zheng*, †, Jiahe Guo, Yuqi Wu, Jie Zhou, Jiwen Lu

清华的工作，500 stars

[website](http://wzzheng.net/StreamVGGT/) [arxiv](https://arxiv.org/pdf/2507.11539) [github](https://github.com/wzzheng/StreamVGGT)

> **StreamVGGT 是一个基于因果 Transformer 的实时 4D 视觉几何重建模型，支持**  
> **“逐帧增量更新 + 缓存历史 token 记忆”，在保持 VGGT 级精度的同时实现低延迟在线推理。**

- **4D 重建**：从视频中恢复动态 3D 场景 + 时间维度，是 CV 的基础任务。
- **现有问题**：
  - **离线模型**（如 VGGT、Fast3R）每次都要重新处理整个序列，无法实时；
  - **流式模型**（如 Spann3R、CUT3R）虽支持在线更新，但存在误差累积；
  - **效率瓶颈**：全局注意力复杂度高，不适合长序列。

**模型结构**

| 模块 | 设计 | 作用 |
|---|---|---|
| **因果 Transformer** | 仅允许当前帧关注历史帧，模拟“人类感知” | 保证因果性，减少计算 |
| **Cached Token Memory** | 缓存历史帧的 key/value token，避免重复计算 | 实现“增量更新” |
| **知识蒸馏** | 用 VGGT（全局注意力）作为教师，蒸馏到因果学生模型 | 抑制误差累积，提升精度 |
| **FlashAttention-2** | 集成高效注意力算子 | 实现实时推理 |

| 模块 | 说明 |
|---|---|
| **Image Encoder** | 使用 DINOv2 提取图像 token |
| **Spatio-Temporal Decoder** | 替换 VGGT 的全局注意力为时间因果注意力 |
| **Cached Memory** | 缓存历史 token，推理时只处理当前帧 |
| **Multi-Task Heads** | 同时输出：<br>• 相机位姿（pose）<br>• 深度图（depth）<br>• 点云图（point map）<br>• 2D 点追踪（tracking） |

**实验结果**

| 任务 | 数据集 | 对比模型 | 结果 |
|---|---|---|---|
| **3D 重建** | 7-Scenes / NRGBD / ETH3D | vs CUT3R、Spann3R | **优于 SOTA 流式模型**，接近 VGGT |
| **单帧深度估计** | KITTI / Sintel / NYU-v2 | vs MonST3R、DUSt3R | **优于所有流式模型** |
| **视频深度估计** | Sintel / Bonn / KITTI | vs CUT3R、Point3R | **优于 CUT3R** |
| **相机位姿估计** | CO3Dv2 | vs VGGT | **AUC@30 达到 82.4**（接近 VGGT 87.7） |

**推理效率**

| 帧数 | VGGT（全局） | StreamVGGT（流式） |
|---|---|---|
| N=1 | 2089 ms | 386 ms |
| N=10 | 2000 ms | 67 ms |
| N=40 | 2089 ms | 68 ms |

> ✅ **StreamVGGT 在长序列下推理时间几乎恒定，VGGT 随帧数线性增长。**

**局限性**

| 问题 | 说明 |
|---|---|
| **内存膨胀** | 缓存 token 随帧数线性增长，不适合超长序列 |
| **教师模型限制** | 蒸馏依赖 VGGT，极端场景（高速、非刚性）表现下降 |
| **部署限制** | 当前模型较大，不适合移动端 |