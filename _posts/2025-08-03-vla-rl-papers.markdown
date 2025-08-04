---
layout: post
title:  "看看论文"
date:   2025-08-03 00:14:00 +0800
categories: posts
tag: autonomous driving
---

# B2DVL完事了，看看最近的论文来搞新东西

## Reinforcement Learning for Flow-Matching Policies  
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

## 拓展：关于 flow matching

有一篇讲 transition matching 的知乎文章： [Transition Matching: Scalable and Flexible Generative Modeling （公式看晕了，喂！） - 知乎](https://zhuanlan.zhihu.com/p/1928847692173390104)


## Steering Your Diffusion Policy with Latent Space Reinforcement Learning

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

一句话带走
DSRL 把“微调大模型”变成“微调小噪声”，让真实机器人用几十次交互就能把 20 % 成功率飙到 90 %，并首次把 RL 成功塞进 3.3 B 参数的通用策略 π0，为现场自适应打开了实用大门。