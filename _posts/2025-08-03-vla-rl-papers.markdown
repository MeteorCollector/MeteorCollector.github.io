---
layout: post
title:  "看看论文"
date:   2025-08-03 00:14:00 +0800
categories: posts
tag: autonomous driving
---

# B2DVL完事了，看看最近的论文

> 一般来讲，越靠上的越新

## Diffusion

### DiWA: Diffusion Policy Adaptation with World Models

Akshay L Chandra1∗, Iman Nematollahi1∗, Chenguang Huang2 Tim Welschehold1, Wolfram Burgard2, Abhinav Valada1 1 University of Freiburg 2 University of Technology Nuremberg

[arxiv](https://arxiv.org/abs/2508.03645)    [github (2~3个月之前开源, 23 stars)](https://github.com/acl21/diwa)    [website](https://diwa.cs.uni-freiburg.de/)

这篇论文叫 **《DiWA: Diffusion Policy Adaptation with World Models》**，它提出了一种全新的方法，**让机器人可以在完全不接触真实环境的情况下，通过“想象”来提升已有的技能**，尤其适用于**基于扩散模型（Diffusion Policy）的机器人控制策略**。

> DiWA 是第一个**完全离线**的扩散策略微调框架，它用一个**世界模型**代替真实环境，让机器人通过“在脑子里练”来提升技能，**不需要任何额外真实交互**，却比传统在线强化学习方法更高效、更安全。

#### DiWA 的核心思路

| 阶段 | 任务 | 方法 |
|------|------|------|
| **1. 世界模型训练** | 学会“想象” | 用大量**无标签的机器人自由探索数据**训练一个**世界模型**，能预测未来状态 |
| **2. 策略预训练** | 学会“模仿” | 用少量专家演示数据预训练一个**扩散策略**（Diffusion Policy） |
| **3. 奖励建模** | 学会“目标” | 用专家数据训练一个**奖励分类器**，判断某个状态是否接近任务成功 |
| **4. 离线微调** | 学会“改进” | 在**世界模型里做强化学习（PPO）**，通过“想象的轨迹”来微调策略 |

#### 实验结果

**✅ 模拟环境（CALVIN benchmark）**
| 方法 | 成功率 | 是否在线交互 | 交互次数 |
|------|--------|--------------|----------|
| 原始扩散策略 | 57.8% | ❌ | 0 |
| **DiWA**（离线微调） | **82.3%** | ❌ | **0** |
| DPPO（在线微调） | 82.3% | ✅ | **250万次** |

**✅ 真实世界（Franka机器人）**
| 任务 | 原始策略 | DiWA微调后 |
|------|----------|-------------|
| 打开抽屉 | 55% → **85%** |
| 关闭抽屉 | 60% → **95%** |
| 推滑块 | 55% → **87%** |

####  技术亮点

| 模块 | 作用 | 技术细节 |
|------|------|-----------|
| **Dream Diffusion MDP** | 把扩散过程建模为强化学习问题 | 把每一步“去噪”看作一个动作，整个世界模型作为环境 |
| **世界模型（World Model）** | 替代真实环境 | 基于DreamerV2架构，支持长时序预测 |
| **奖励分类器** | 代替真实奖励 | 用专家数据训练一个二分类器，判断状态是否成功 |
| **行为克隆正则化** | 防止“钻空子” | 限制策略不要太偏离原始行为，避免利用世界模型的缺陷 |



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



### Reinforcement Learning with Action Chunking

[arxiv](https://arxiv.org/abs/2507.07969) 未有开源代码

Qiyang Li, Zhiyuan Zhou, Sergey Levine. UC Berkeley

我对 RL 还并不内行，之后再细看

这篇论文《Reinforcement Learning with Action Chunking》提出了一种**简单但高效的强化学习方法（Q-chunking）**，专门解决**长周期、稀疏奖励任务**在**离线到在线（offline-to-online）RL**场景下的**探索困难和样本效率低**的问题。

#### ✅ 一句话总结：
> **Q-chunking 把“动作序列”当作一个整体来训练和探索，使得 RL 在稀疏奖励任务中能更快、更稳地从离线数据中学到东西，再在线微调。**

#### 🎯 核心思想：动作分块（Action Chunking）

| 传统 RL | Q-chunking |
|--------|------------|
| 每一步预测一个动作 | 每次预测一个**动作序列（chunk）**，比如未来 5 步 |
| 探索靠随机扰动 | 探索靠**模仿离线数据中的连贯行为**，更高效 |
| 1 步 TD 更新慢 | 使用**无偏的 n 步 TD 更新**，更快传播价值信号 |

#### 💡 关键技术点

| 模块 | 说明 |
|------|------|
| **动作空间扩展** | 把动作从 `a_t` 变成 `a_t:t+h`，即一个长度为 h 的序列 |
| **行为约束** | 用 Flow Matching 学一个行为策略，限制策略不过度偏离离线数据 |
| **无偏 n-step TD** | 用完整的动作序列做 TD 更新，避免传统 n-step 的偏差问题 |
| **两种实现** | QC（基于 best-of-N 采样）和 QC-FQL（基于 Flow Q-learning） |

#### 🧪 实验结果

| 基准 | 表现 |
|------|------|
| **OGBench（5 个任务）** | Q-chunking 在最难的 cube-quadruple 上远超所有基线 |
| **Robomimic（3 个任务）** | 在 lift/can/square 上均优于 RLPD、FQL 等 |
| **样本效率** | 仅用 100 万步在线训练就能解决原本几乎学不到的任务 |
| **探索质量** | 动作更连贯，覆盖更多状态空间，避免“原地抖动” |


> **Q-chunking 通过“动作序列预测 + 行为约束 + n-step TD”，让 RL 在稀疏奖励任务中更快、更稳地从离线数据中学到策略，适用于机器人操作等长周期任务。**



### Breaking Imitation Bottlenecks: Reinforced Diffusion Powers Diverse Trajectory Generation

Ziying Song, Lin Liu, Hongyu Pan, Bencheng Liao, Mingzhe Guo, Lei Yang, Yongchang Zhang, Shaoqing Xu, Caiyan Jia, Yadan Luo

[arxiv](https://www.arxiv.org/abs/2507.04049) 暂无开源

> dp + rl，好文。多条轨迹候选我觉得也是一个可行的方式，像之前张兆翔那篇也是这么搞的，预制几个然后diffusion出来实际的。不过他搞了 diffusion policy + rl,领先一步,而且做得非常扎实.应当多看

> 这篇论文提出 **DIVER** —— 首个用 **扩散模型 + 强化学习** 的端到端自动驾驶框架，专门解决传统模仿学习中“**模式崩溃、轨迹单一、过于保守**”的顽疾，让车辆真正生成**多样化且安全可行**的未来轨迹。


核心痛点  
• 现有方法只用一条专家轨迹做模仿，导致：  
  – 所有预测轨迹都“挤”在专家轨迹附近 → 模式崩溃  
  – 不敢变道/超车，无法应对复杂场景  


DIVER 的解法（三步走）

1. Policy-Aware Diffusion Generator (PADG)  
   ‑ 同时输入：地图、周边车辆、**多条参考轨迹**（变道、让行、超车…）  
   ‑ 扩散模型一次性生成 M 条多样轨迹，而非单条均值轨迹  

2. 强化学习“纠偏”  
   ‑ 把扩散过程视为**随机策略**  
   ‑ 用 GRPO 优化**多样性奖励 + 安全性奖励**：  
     • 多样性：轨迹间距离越大越好  
     • 安全：离障碍物越远越好  

3. 新指标 & 训练损失  
   ‑ 提出 **Diversity Metric**（[0,1] 范围）专门衡量多模态轨迹离散度  
   ‑ 用 **匈牙利匹配损失** 将每条预测轨迹对齐到不同参考意图，避免收敛到同一均值  


实验结果  
• **Bench2Drive 闭环**：成功率 ↑29 %、超车能力 ↑3.8 %、紧急制动 ↑5.5 %  
• **NuScenes 开环**：平均多样性指标 ↑61 %，碰撞率 ↓12.5 %  
• 在**转弯、对抗、雨雪雾**等子集上均优于 VAD、SparseDrive、DiffusionDrive 等 SOTA  

DIVER 把“扩散去噪过程”当成一个**可学习的随机策略**（policy），然后用 **Group Relative Policy Optimization（GRPO）** 做 RL 微调，核心流程如下：

#### rl 做 dp 详细方法

**1. 把扩散模型当策略**
- 扩散模型每一步的 denoising 网络 `εθ(τt, t)` 就是**策略 πθ**  
- 输入：当前噪声轨迹 `τt` + 场景条件  
- 输出：下一步去噪方向（相当于动作）

**2. 设计两条 reward 信号** 
| 奖励 | 公式（直觉） | 作用 |
|---|---|---|
| **多样性奖励 rdiv** | `1/M(M-1) Σ‖τi − τj‖₂` | 最大化轨迹间距离，防止模式崩溃 |
| **安全奖励 rsafe** | `−1/T Σ 𝟙[Dsafety(xt) < dthresh]` | 离障碍物越近惩罚越大，保证物理可行 |

**总奖励**：`r(τ) = rdiv + λ·rsafe`

**3. 用 GRPO 更新策略**
- 对每个场景采样 **G 条轨迹**（group）  
- 计算组内 **相对优势**（relative advantage）：  
  `Ai = ri − mean(rgroup)`  
- GRPO 目标：  
  `JGRPO = E[min(wi Ai, clip(wi,1±ε) Ai)] − β·DKL(πθ‖πref)`  
  `wi = πθ/πold`（重要性采样比）

**4. 训练损失组合**
`Ltotal = λmatch·Lmatch + λRL·LRL`

- `Lmatch`：匈牙利匹配损失，保证每条预测轨迹对齐不同意图  
- `LRL`：GRPO 损失，直接用不可微的 `rdiv + rsafe` 优化扩散策略

#### 结果  
- 无需手工标签，仅靠奖励即可让扩散模型**“敢变道”**、**“懂避障”**  
- 在闭环 Bench2Drive 上：成功率 ↑29 %，多样性 ↑66 %，碰撞率 ↓12 %

> **DIVER 把扩散去噪视为策略，用 GRPO + diversity/safety reward 直接微调，突破了模仿学习的“单专家”枷锁。**



### DYNA: Reinforcement Learning in Real World 

[Research - DYNA Robotics | Research](https://www.dyna.co/dyna-1/research) 这个不是论文,,,



## VLA

### FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation

Cui Miao1 TaoChang1 MeihanWu1 HongbinXu2 Chun Li3 MingLi4* Xiaodong Wang1 1 National University of Defense Technology 2Bytedance Seed 3Shenzhen MSU-BIT University 4Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)

[arxiv](https://arxiv.org/abs/2508.02190) 

> 分布式训练 VLA，现在我这边可能不太需要吧，先放在这

这篇文章提出了一种名为 **FedVLA** 的联邦学习框架，用于在机器人操作任务中训练 **视觉-语言-动作（VLA）模型**，同时保护用户隐私。文章的核心贡献是解决了**如何在分布式环境中高效训练多模态机器人模型**，同时**避免集中式训练带来的隐私风险**。

> FedVLA 是第一个用于机器人操作任务的联邦视觉-语言-动作学习框架，它通过“任务感知特征提取 + 双向专家选择 + 专家驱动的聚合策略”，在保护隐私的同时实现了接近集中式训练的性能。

#### FedVLA 的三大创新模块

| 模块名称 | 作用 | 关键技术 |
|----------|------|-----------|
| **IOSP**（Instruction-Oriented Scene-Parsing） | 把图像分解为“任务相关”的对象表示 | 用CLIP模型将图像中的目标物体、周围物体、背景物体与语言指令对齐 |
| **DGMoE**（Dual Gating Mixture-of-Experts） | 让模型根据任务复杂度动态选择专家，提高计算效率 | 引入“**自感知专家**”，专家可以主动决定是否接收token，实现双向选择 |
| **EDA**（Expert-Driven Aggregation） | 在联邦聚合时，优先合并“专家选择相似”的客户端模型 | 利用专家激活向量计算客户端之间的相似度，动态分配聚合权重 |

#### 实验验证

**✅ 模拟环境（Meta-World）**
- 任务：关门、关抽屉、扫地、开窗
- FedVLA 成功率：**63.3%**（vs 集中式 65.0%，FedAvg 51.7%）

**✅ 真实世界（UR3机械臂）**
- 任务：清理桌面、扔垃圾、开抽屉、分药
- FedVLA 成功率：**63.3%**（vs 集中式 63.4%，FedAvg 53.3%）



### VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting

[arxiv](https://arxiv.org/abs/2507.05116) [github](https://github.com/LukeLIN-web/VOTE 仅几天前开源，16stars)

> **VOTE 提出“单令牌动作 + 投票集成”的轻量级 VLA 框架，把生成动作所需的 token 数量从上百个压缩到 1 个，同时在推理阶段用历史预测投票纠错，实现 39× 推理加速、98% LIBERO 成功率，并能在边缘设备 46 Hz 实时运行。**

| 传统 VLA 模型 | 问题 |
|---|---|
| 多 token / 扩散动作头 | **推理慢**：OpenVLA/CogACT 每步 200 ms 以上 |
| 只执行当前预测 | **动作浪费**：上一帧预测被丢弃，轨迹抖动 |
| 3D/扩散增强 | **训练贵**：显存大、梯度步数多 |


#### 1️⃣ 训练阶段：单令牌动作  
- 在 LLM tokenizer 内新增 `<ACT>` 特殊 token  
- **一次前向只生成 1 个 token** → 隐藏态直接输入轻量 MLP 动作头  
- **token 数从 N×D ↓ 1**（N=chunk, D=action dim）  
- LoRA+瓶颈 MLP，训练步数≈OpenVLA-OFT 的 **15–54 %**  

#### 2️⃣ 推理阶段：Trajectory Ensemble Voting  
- 维护一个 **K+1 步动作委员会**（历史+当前）  
- 用余弦相似度投票，**选多数派平均值**作为最终动作  
- **τ=0.5** 经验阈值即可，无需调参  
- 融合后轨迹更平滑，平均提升 **5–10 % SR**

| 场景 | 数据量 | 成功率 | 延迟/吞吐 | 边缘设备 |
|---|---|---|---|---|
| **LIBERO-4 套件** | 50–130K 步 | **98 %** | 78 ms / 102 Hz | 46 Hz @Jetson Orin |
| **SimplerEnv-WidowX** | 60K 步 | **58 %** | 78 ms | 实时 |
| **Google Robot** | 150K 步 | **74 %** | 78 ms | 实时 |

- **比 OpenVLA 快 39×**，显存占用 **-50 %**  
- **比 CogACT 高 7 % SR**，边缘设备 CogACT OOM

> **VOTE 用“1 个 token 生成整个动作块 + 历史投票纠错”，让 VLA 模型在边缘设备上也能以 46 Hz 的实时速度完成高成功率机器人操作。**

#### 训练用数据

**1️⃣ 数据模态**
| 模态 | 说明 |
|------|------|
| **RGB 图像** | 1 张 224×224 第三人称相机图像（或更多视角可扩展） |
| **语言指令** | 自然语言任务描述（如 “put the cup on the table”） |
| **动作标签** | 连续 7-DoF 末端位姿 + 1 维夹爪开闭（共 8 维） |
| **动作块（chunk）** | 连续 N 步动作序列（N=8 或 16，按实验设定） |

> **无需深度图、点云、3D 姿态等额外模态**，保持极简输入。


**2️⃣ 数据量（按任务/场景）**
| 场景 | 单任务数据量 | 总数据量 | 备注 |
|------|---------------|-----------|------|
| **LIBERO 基准** | 50–130K 步 | 4 个任务 × 50–130K ≈ **200–520K 步** | 每个任务 500 条演示 × 100 步 |
| **SimplerEnv** | 60K 步 | 1 个任务 ≈ **60K 步** | BridgeDataV2 子集 |
| **真实机器人** | 类似量级 | 视任务而定 | 论文未明确，但经验值 **50–200 条演示/任务** |

> **数据量远小于 OpenVLA-OFT**（后者需 150K 步 × 64 batch）。

**计算资源**
| 阶段 | 硬件需求 | 训练时间 | 显存占用 | 备注 |
|------|-----------|-----------|-----------|------|
| **微调 VLA（LoRA）** | 2×H100 (94GB) | 8–130K 步（1–3 天） | 14–19 GB | 全局 batch=40，LoRA rank=32 |
| **边缘推理** | NVIDIA Jetson AGX Orin (32 GB) | 实时 46 Hz | **346 ms 延迟** | 无需再训练 |
| **对比基线** | 8×A100 (80GB) | 150K 步 × 64 batch | 19 GB | OpenVLA-OFT 资源 |


> **VOTE 仅需 RGB + 语言 + 动作轨迹，单任务几十到几百条演示，2×H100 上 1–3 天即可微调完成，边缘设备实时运行。**



### Video Generators are Robot Policies

Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, Carl Vondrick

[arxiv](https://arxiv.org/abs/2508.00795) 未开源

> 不看好非要生成视频的模型，跑不起来... 8*A100 练两周真绷不住了

> **本文提出“Video Policy”：用大规模视频生成模型当“世界模拟器”，再配一个轻量级动作解码器，就能把生成视频直接变成机器人策略，实现**少量动作数据、强鲁棒性的泛化控制**。

#### 📌 核心思路：把“视频生成”当策略  
1. **先视频**：用 **Stable Video Diffusion (SVD)** 级联模型，根据 **初始图像 + 语言任务** 生成未来机器人执行视频帧。  
2. **后动作**：用 **轻量级 1D-CNN U-Net** 从视频隐藏层特征解码 **7-DoF 末端位姿 + 夹爪开闭** 的连续动作序列。  

> 因为 SVD 已在互联网海量视频里学到通用动力学，**动作解码器只需 50~200 条演示即可泛化**。

#### 📌 训练与数据  
| 阶段           | 数据                                       | 计算资源        | 说明                     |
| -------------- | ------------------------------------------ | --------------- | ------------------------ |
| **视频预训练** | 互联网通用视频                             | 8×A100，两周    | 微调 SVD                 |
| **动作微调**   | 每任务 **50 条人演示**（或 300M MimicGen） | 同上            | 冻结视频权重，只训动作头 |
| **真实世界**   | 200 条/任务（5 任务）                      | RTX 4070 Laptop | 实时 30 步扩散推理       |

#### 📌 主要结果  
| Benchmark            | 数据量   | 平均成功率 | 对比                        |
| -------------------- | -------- | ---------- | --------------------------- |
| **RoboCasa-24 任务** | 50 demo  | **66%**    | 超 DP-ResNet、UVA、GR00T 等 |
| **LIBERO-10**        | 50 demo  | **94%**    | 显著优于 UVA                |
| **真实 5 任务**      | 200 demo | 0.3-1.0    | 对未见物体/背景/位置均泛化  |

### 📌 关键发现  
1. **2-阶段训练 > 联合训练**：先训视频再训动作，效果更好（63% vs 57%）。  
2. **越长视频预测越泛化**：32 步视频预测在分布偏移任务上收益最大。  
3. **无动作视频也能泛化**：仅动作头训 12/24 任务，靠视频先验仍超过基线。  
4. **实时瓶颈**：25 帧 256×256 在 A100 上需 ~9 秒，但加速技术可解。



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



### TriVLA: A Triple-System-Based Unified Vision-Language-Action Model for General Robot Control

[arxiv](https://arxiv.org/abs/2507.01424) [website](https://zhenyangliu.github.io/TriVLA/)

> 只有网站，还没有开源相关代码。这个多段其实比较符合我的想法，我觉得端到端直接来是真的不好搞啊。但是这个训练得实在是太耗资源了，要用视频模型预测未来动态场景，直接 8 个 H100 练 2-3 天，太凶猛了。我不看好用视频模型预测未来，没有必要。

这篇论文《TriVLA: A Triple-System-Based Unified Vision-Language-Action Model for General Robot Control》提出了一种**新的三系统统一架构**，用于提升机器人在**动态环境中执行长周期、复杂指令任务**的能力。

**核心创新：三系统架构（Triple-System）**

| 系统编号 | 名称 | 作用 | 技术实现 |
|----------|------|------|----------|
| **System 2** | Vision-Language Module | 理解语言指令 + 场景语义 | 使用预训练 Eagle-2 VLM（SmolLM2 + SigLIP-2） |
| **System 3** | Dynamics Perception Module | 预测未来动态场景（视频级） | 微调 Stable Video Diffusion（SVD）模型 |
| **System 1** | Policy Learning Module | 生成连续动作序列 | 使用 Diffusion Transformer + Flow Matching |

**解决的问题**

| 传统方法问题 | TriVLA 如何解决 |
|--------------|------------------|
| 只看当前图像，忽略动态变化 | System 3 预测未来帧，建模动态 |
| 缺乏语言与视觉的深度对齐 | System 2 使用预训练 VLM 处理语言和图像 |
| 动作生成不连贯、频率低 | System 1 使用扩散模型生成动作 chunk，支持 36Hz 控制频率 |

**实验结果**

| 场景 | 数据集 | 表现 |
|------|--------|------|
| **仿真** | CALVIN ABC→D | 平均任务长度 **4.37**（SOTA） |
| **仿真** | MetaWorld（60任务） | 平均成功率 **71.4%**（优于 VPP、GR-1 等） |
| **仿真** | LIBERO（4套件） | 在 Spatial/Object/Goal/Long 任务中均领先 |
| **真实机器人** | Franka/Kinova/Fair | 在少量演示下成功完成长周期任务 |

**关键亮点**

- **数据效率高**：仅用 10% CALVIN 数据，性能优于全数据训练的 GR-1。
- **控制频率高**：36Hz 实时控制，优于传统扩散策略。
- **通用性强**：支持不同机器人（Franka、Kinova、Fair）和多视角输入。
- **长周期任务能力强**：可处理多步指令，如“打开抽屉→取出方块→放入盒子→关闭抽屉”。

#### 所需数据

| 模态 | 内容 | 格式 |
|------|------|------|
| **视觉** | 多视角 RGB 图像（静态摄像头 + 手腕摄像头） | PNG/JPEG |
| **语言** | 自然语言任务指令 | 文本 |
| **动作** | 连续 7 维动作（末端位姿 + 夹爪状态） | NumPy 数组 |
| **状态** | 机器人关节角度、末端位姿、速度等 | NumPy 数组 |
| **视频** | 完整操作视频序列（用于 System 3） | MP4 |
| **人类操作视频** | 互联网人类操作视频（用于预训练） | MP4 |
| **辅助标签** | 可选：深度图、物体掩码、关键点等 | NumPy 数组 |

| 数据类型 | 来源 | 量级 |
|----------|------|------|
| **人类操作视频** | Something-Something V2、YouTube 等 | **193,690 条** |
| **机器人操作视频** | Open X-Embodiment、DROID、CALVIN | **179,074 条** |
| **任务演示视频** | CALVIN、MetaWorld、LIBERO、真实机器人 | **每任务 50–100 条** |
| **微调数据** | 真实机器人（Franka/Kinova/Fair） | **每任务 100 条轨迹** |

#### 三阶段训练流程

**阶段 1：System 3 视频扩散模型微调（VDM）**
- **目标**：让视频模型学会预测未来帧（动态感知）
- **数据**：
  - 人类操作视频（193k）
  - 机器人操作视频（179k）
- **训练设置**：
  - 模型：Stable Video Diffusion（1.5B 参数）
  - 损失：扩散重建损失
  - 训练时间：**2–3 天，8×H100 GPU**
  - 冻结参数：训练完成后冻结 System 3

**阶段 2：System 1 策略网络训练（Policy Learning）**
- **目标**：学习从视觉+语言+动态表示 → 动作的映射
- **数据**：
  - CALVIN、MetaWorld、LIBERO、真实机器人演示
- **训练设置**：
  - 模型：Diffusion Transformer（DiT）
  - 损失：Flow Matching + MSE 动作损失
  - 训练时间：**5–9 小时，4×H100 GPU**
  - 控制频率：**36 Hz**
  - 动作 chunk 长度：10 步

**阶段 3：System 2 VLM 微调（可选）**

- **目标**：增强语言理解能力（已在 Eagle-2 中预训练）
- **数据**：
  - 任务指令 + 图像对
- **训练设置**：
  - 模型：Eagle-2（SmolLM2 + SigLIP-2）
  - 冻结主干，仅微调 LoRA 层
  - 输入：224×224 图像 + 文本指令
  - 输出：第 12 层 token（用于策略输入）



### Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation

Wenbo Zhang, Tianrun Hu, Yanyuan Qiao, Hanbo Zhang, Yuchu Qin, Yang Li, Jiajun Liu, Tao Kong, Lingqiao Liu, Xiao Ma

主要是字节的工作

[arxiv](https://arxiv.org/abs/2506.09990) [github (上个月开源, 54 stars)](https://github.com/ByteDance-Seed/Chain-of-Action)

> 说是比 diffusion policy 要好, 真的假的

这篇论文提出了一种新的机器人操作策略范式 —— **Chain-of-Action（CoA）**，全称是“链式动作轨迹自回归建模”，用于解决传统机器人策略在长时任务中**误差累积严重、空间泛化能力差**的问题。

#### 📌 研究背景
- **传统方法的问题**：大多数机器人策略采用“正向预测”方式，即根据当前观察逐步预测下一步动作。这种方式容易在长时间任务中累积误差，导致任务失败。
- **核心挑战**：如何让机器人策略具备**全局任务目标感知的推理能力**，从而提高长时任务的成功率和空间泛化能力。

#### 📌 核心创新：Chain-of-Action（CoA）
CoA 的核心思想是**反向推理动作序列**：
- **从任务目标（关键帧动作）出发**，逆向生成整个动作轨迹。
- 这种“从目标倒推”的方式天然具有**全局到局部（global-to-local）**的结构，每一步动作都被最终目标所约束，从而显著减少误差累积。

#### 📌 四大关键设计
为实现上述反向建模，CoA 引入了四项必要机制：

| 设计组件 | 作用 |
|----------|------|
| **连续动作表示** | 避免离散化带来的精度损失，适合精细控制。 |
| **多Token预测（MTP）** | 同时预测未来多个动作，增强局部一致性。 |
| **动态停止机制** | 根据当前状态与目标的接近程度自动决定轨迹长度。 |
| **反向时间集成（Reverse Temporal Ensemble）** | 多次反向生成轨迹并集成，提升鲁棒性。 |

#### 📌 实验结果
**✅ 模拟环境（RLBench，60个任务）**
- CoA 相比 ACT（Transformer策略）**提升16.3%**，相比 Diffusion Policy **提升23.2%**。
- 在**空间泛化能力**上表现尤为突出，尤其在物体位置变化大的任务中优势明显。

**✅ 真实世界（8个厨房任务，Fetch机器人）**
- 成功率达 **61.3%**，明显高于 ACT（46.3%）和 Diffusion Policy（36.3%）。
- 验证了 CoA 在实际部署中的有效性。

训练过程分为 **模拟实验（RLBench）** 和 **真实机器人实验** 两部分。

**✅ 1. 模拟实验（RLBench）**

| 项目         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据来源** | RLBench 专家演示（Franka Panda 7-DoF 机器人，桌面任务）      |
| **数据内容** | 多视角 RGB 图像（128×128），夹爪状态，动作轨迹               |
| **数据量**   | 每任务 **100 条专家演示**（仅使用 variation 0 以节省计算）   |
| **总任务数** | 60 个任务（对比 ACT/DP）；10 个任务（对比 Octo）             |
| **总数据量** | 约 **6,000 条演示**（60×100）                                |
| **计算资源** | **1× NVIDIA H100 GPU**（单卡）                               |
| **训练时间** | **20,000 步**，batch size = 128，约 **1～2 天**              |
| **网络结构** | ResNet-18 + 4层 Transformer Encoder + 7层 Transformer Decoder |


**✅ 2. 真实机器人实验（Fetch 机器人）**

| 项目         | 说明                                                |
| ------------ | --------------------------------------------------- |
| **数据来源** | 8 个厨房操作任务（如开抽屉、放杯子、关微波炉等）    |
| **数据内容** | 单视角 RGB（640×480 → 224×224），夹爪状态，动作轨迹 |
| **数据量**   | 每任务 **35～81 条专家演示**                        |
| **总数据量** | 约 **400～600 条演示**（8×50 平均值）               |
| **计算资源** | **1× NVIDIA RTX 4070 Laptop GPU**（实时推理）       |
| **训练时间** | 与模拟设置一致，约 **1～2 天**                      |
| **推理频率** | 策略 10Hz，PD 控制器 1000Hz                         |

**✅ 总结表格**

| 场景             | 数据量（每任务） | 总数据量  | 计算资源       | 训练时间 |
| ---------------- | ---------------- | --------- | -------------- | -------- |
| **RLBench 模拟** | 100 条           | ~6,000 条 | 1× H100        | ~1–2 天  |
| **真实机器人**   | 35–81 条         | ~500 条   | 1× 4070 Laptop | ~1–2 天  |

> **CoA 每任务仅需几十到上百条专家演示，单张 GPU（H100 或 4070）即可在 1～2 天内完成训练，资源需求远低于视频生成类方法。**

> **Chain-of-Action 通过“从目标倒推”的反向动作建模方式，显著提升了机器人策略在长时、复杂任务中的鲁棒性和泛化能力，是一种新的、有前景的机器人动作建模范式。**



### Efficient Robotic Policy Learning via Latent Space Backward Planning

Dongxiu Liu, Haoyi Niu, Zhihao Wang, Jinliang Zheng, Yinan Zheng, Zhonghong Ou, Jianming Hu, Jianxiong Li, Xianyuan Zhan

[arxiv](https://arxiv.org/abs/2505.06861) [website](https://lbp-authors.github.io/)

> 主页不错, 可以fork. 非常神秘, 和coa一样, 这一篇也是反向操作. 不过训练耗费资源还是相当大的.

这篇文章提出了一种**高效、鲁棒的机器人长时任务规划框架**，名为 **LBP（Latent Space Backward Planning）**，核心思想是：

> **“从最终目标出发，反向递归地规划中间子目标，从而在长时任务中实现高效、准确、任务一致的规划。”**

#### 研究背景与问题
当前机器人规划方法面临一个“效率-精度-一致性”三难困境：

| 方法类型 | 优点 | 缺点 |
|----------|------|------|
| **细粒度视频预测**（如UniPi、GR-1） | 提供丰富未来信息 | 计算昂贵、误差累积严重 |
| **粗粒度子目标规划**（如GCSL、SuSIE） | 计算轻量 | 正向规划容易偏离最终目标，导致“跑题” |

#### 📌 LBP 的核心创新

**✅ 1. 潜空间建模（Latent Space）**
- **不在像素空间做规划**，而是在**视觉潜空间**（如DecisionNCE、SigLIP）中进行，大幅降低计算量。
- 通过潜空间保留语义信息，同时避免高维图像带来的冗余。

**✅ 2. 反向规划（Backward Planning）**
- **从最终目标出发**，递归地生成中间子目标，逐步靠近当前状态。
- 每一步子目标都**与最终目标对齐**，避免“跑偏”。

**✅ 3. 子目标融合（Goal-Fusion Attention）**
- 使用 **Perceiver-style Cross-Attention** 动态融合不同距离的子目标信息。
- 让策略在不同阶段**自适应地关注短期 vs 长期信息**。

| 步骤 | 说明 |
|------|------|
| **Step 1：目标生成** | 根据当前图像 + 语言指令，预测最终潜空间目标（latent goal） |
| **Step 2：子目标反向生成** | 从目标开始，递归生成中间子目标（由远到近） |
| **Step 3：策略执行** | 用子目标序列作为上下文，训练或引导策略执行动作 |

#### 📌 实验结果

✅ LIBERO-LONG 模拟基准（10个长时任务）
| 方法 | 平均成功率 |
|------|-------------|
| LBP（DecisionNCE） | **88.6%** ✅ |
| Seer | 78.6% |
| SuSIE | 76.3% |
| OpenVLA | 54.0% |

> LBP 显著优于所有基线，尤其在多阶段任务中表现突出。

✅ 真实机器人实验（4个长时任务）
| 任务 | 阶段数 | LBP 平均得分 |
|------|--------|---------------|
| Move cups | 2 | 77.9 ✅ |
| Stack 4 cups | 3 | 72.5 ✅ |
| Shift cups | 5 | 67.1 ✅ |

> LBP 在**后期阶段**优势显著，说明其**长时一致性更好**。
> **LBP 用“从终点倒推”的方式，在潜空间中轻量、准确地规划子目标，显著提升了机器人长时任务的成功率和泛化能力。**

#### 训练过程

✅ 1. **高层规划器（High-Level Planner）**
> 负责生成最终目标（latent goal）和子目标（subgoals）

| 项目         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **所需数据** | 专家演示视频（含图像 + 语言指令 + 动作）                     |
| **数据量**   | 每个任务 **50条专家演示**（LIBERO-LONG）<br>每个真实任务 **200条专家演示** |
| **数据格式** | 图像（多视角）、语言指令、动作序列                           |
| **计算资源** | 1× NVIDIA H100 GPU<br>训练 100k steps，batch size = 64       |
| **训练时间** | 约 **2~3天**（单卡）                                         |


✅ 2. **低层策略（Low-Level Policy）**
> 根据子目标序列执行动作

| 项目         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **所需数据** | 与高层共用（专家演示）                                       |
| **数据量**   | 同上（50/200条）                                             |
| **计算资源** | 1× NVIDIA H100 GPU（模拟）<br>1× NVIDIA 4070 Laptop GPU（真实机器人） |
| **训练时间** | 模拟：200k steps（约3~4天）<br>真实机器人：400k steps（约5~6天） |



| 模块       | 数据量（每任务）        | 计算资源                      | 训练时间（单卡） |
| ---------- | ----------------------- | ----------------------------- | ---------------- |
| 高层规划器 | 50（模拟）/ 200（真实） | 1× H100                       | ~2~3天           |
| 低层策略   | 同上                    | 1× H100（模拟）/ 4070（真实） | 3~6天            |

---

> **每任务只需几十到几百条专家演示视频，单张GPU即可在几天内完成训练，资源门槛远低于视频生成类方法。**

如需更低资源版本（如更少数据或更小模型），作者也指出LBP支持灵活压缩子目标数量（如只预测2~3个），可进一步降低训练成本。






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


## 视频生成

### StreamDiT: Real-Time Streaming Text-to-Video Generation

Akio Kodaira, Tingbo Hou, Ji Hou, Masayoshi Tomizuka, Yue Zhao

UC Berkeley, Meta

[arxiv](https://arxiv.org/abs/2507.03745) [website](https://cumulo-autumn.github.io/StreamDiT/)

> 囤在这里,虽然不做视频生成,但是还是要看

这篇文章提出了一种名为 **StreamDiT** 的新型实时流式文本到视频生成模型，旨在解决现有视频生成模型只能离线生成长度有限、短时视频片段的问题。文章的核心贡献和创新点可以总结为以下几点：

#### 研究背景与问题
- **现有问题**：虽然基于扩散变换器（DiT）的文本到视频（T2V）模型在视频质量上取得了显著进展，但它们通常只能离线生成短片段，难以满足实时、交互式应用的需求。
- **关键挑战**：如何在保证视频内容一致性和高质量的前提下，实现**流式、实时、长视频**的生成。

#### 核心贡献

1. 提出 **StreamDiT 训练框架**
- **基于流匹配（Flow Matching）** 的训练方法，引入了一个**滑动缓冲区（moving buffer）** 来处理视频帧序列。
- **统一的分区策略（partitioning scheme）**：将缓冲区中的帧划分为多个“块”（chunks），每个块包含多个帧和微步（micro-steps），从而统一了传统扩散模型中的均匀噪声和FIFO-Diffusion中的对角线噪声方法。
- **混合训练策略**：在训练中同时使用多种分区方案（如不同的块大小和步数），增强了模型的泛化能力，避免了过拟合某一特定方案。

2. 设计 **高效的 StreamDiT 模型架构**
- **基于 adaLN DiT（自适应层归一化的扩散变换器）**，引入了**可变时间嵌入（varying time embedding）** 和**窗口注意力（window attention）**，以提升效率并适应流式生成。
- **模型规模**：训练了一个 40 亿参数（4B）的模型，能够在单张 H100 GPU 上实现 16FPS 的实时生成。

3. 提出 **多步蒸馏（Multistep Distillation）方法**
- 针对 StreamDiT 的特殊分区设计，定制了蒸馏策略，将原本需要 128 步的去噪过程压缩到 8 步，同时不依赖无分类器引导（CFG），实现了实时推理。

#### 实验
- **与现有方法对比**：在 VBench 和人类评估中，StreamDiT 在长视频生成的质量、一致性、动态性等方面均优于 ReuseDiffuse 和 FIFO-Diffusion 等现有流式生成方法。
- **消融实验**：验证了混合训练策略的有效性，表明混合不同分区方案能提升生成质量。
- **应用场景**：
  - **实时流式生成**：可实时生成长达 1 分钟以上的视频。
  - **交互式生成**：用户可实时输入提示词，动态改变视频内容。
  - **视频到视频编辑**：支持实时视频编辑任务（如将视频中的猪变成猫）。


> **StreamDiT 提出了一种新颖的流式文本到视频生成框架，通过创新的训练策略、高效模型架构和定制蒸馏方法，首次实现了在单张 GPU 上的实时、高质量、长视频生成，为交互式视频应用开辟了新可能。**





## 其他

### DenseMixer: Improving MoE Post-Training with Precise Router Gradient

**Feng Yao$^{\star\dagger}$      Junxia Cui$^{\star}$      Ruohan Zhang$^{\star}$      Liyuan Liu$^{\dagger}$      Shibo Hao      Li Zhang      Chengyu Dong      Shuohang Wang      Yelong Shen      Jianfeng Gao      Jingbo Shang**

$^{\dagger}$: Project Lead; $^{\star}$: Core Contributors; (Work in Progress)

**UCSD,  Microsoft**

[blog](https://fengyao.notion.site/moe-posttraining) [github 两个月前开源，star 58](https://github.com/yaof20/DenseMixer)

> MoE 的优化工作

这篇文章主要介绍了一个新方法 **DenseMixer**，旨在改进 **Mixture-of-Experts（MoE）** 模型的后训练（post‑training）效果。核心内容如下：

1. **研究背景与现有挑战**
   MoE 模型相比稠密模型来说训练更困难，关键问题源于其稀疏路由机制（Top‑K router），该机制非可微，导致梯度反向传播复杂。

2. **DenseMixer 方法**
   论文提出了 DenseMixer 技术，通过在前向传播时对所有专家（包括未激活的专家）进行计算，来获取更加精确的路由梯度。它通过多付出一次前向计算的代价，换取更优的梯度估计。

3. **适应性强，使用简单**

   * **兼容性广**：支持不同规模的 MoE（如 7B、14B、30B）、架构（是否共享专家）、预训练方式（从零开始或 “up‑cycling”）、以及不同后训练数据类型（如 instruction tuning 或 long chain-of-thought 数据）。
   * **方便使用**：只需执行 `pip install densemixer` 然后 `densemixer setup`，设置环境变量 `DENSEMIXER_ENABLED=1` 即可启用 DenseMixer，无需代码更改、推理无额外开销。

4. **实验表现优异**
   在多个任务的多种模型规模下，DenseMixer 均显著优于传统方法，平均提升一般在 **2% 左右**，在某些评测上甚至更高。


总结来说，这篇文章提出的 **DenseMixer** 是一种简单易用、兼容性强且效果可靠的 MoE 后训练方案，通过改善梯度估计质量来提升模型表现。



