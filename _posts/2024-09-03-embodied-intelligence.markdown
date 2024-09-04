---
layout: post
title:  "具身智能一瞥"
date:   2024-09-03 15:57:00 +0800
categories: posts
tag: embodied ai
---

## 前情提要

“具身智能的切入点是灵活手”

为了明确未来的研究方向，需要对具身智能有一定了解。毕竟现在看来cv的未来还是与robotics结合的。

具身智能和自动驾驶差不多，最传统的思路还是模仿学习和强化学习。作为非ai专业学生，还是要补一补这方面的知识。

有两个比较好的学习资料，放在这里，以后要找时间啃下来：

LAMDA的模仿学习教程：[Imitation_Learning.pdf (nju.edu.cn)](https://www.lamda.nju.edu.cn/xut/Imitation_Learning.pdf)

上交“强化学习”课程课本《动手学强化学习》：[前言 (boyuai.com)](https://hrl.boyuai.com/chapter/intro)

为了知道大家都在干啥，还是第一时间读一些论文。

## Definition from CMU

***Robots that perceive, act and collaborate.***

Embodied AI is the integration of machine learning, computer vision, robot learning and language technologies, culminating in the “embodiment” of artificial intelligence: robots that can perceive, act and collaborate.

Robotics, Embodied AI and Learning (REAL) hosts a team of researchers and scientists who are elevating the role of robots from tools to partners, working side by side with humans to solve complex real-world challenges.

## Description from ACM

These foundational studies highlight **three principles** for developing EAI (Embodied Artificial Intelligence) systems. First, EAI systems must not rely on predefined, complex logic to manage specific scenarios. Second, it is essential that EAI systems incorporate evolutionary learning mechanisms, enabling them to adapt continuously to their operational environments. Lastly, the environment plays a pivotal role in shaping not just physical behaviors, but also cognitive structures.

https://cacm.acm.org/blogcacm/a-brief-history-of-embodied-artificial-intelligence-and-its-future-outlook/

## Do As I Can, Not As I Say: Grounding Language in Robotic Affordances

这是比较早期（2022）年的工作 SayCan，主要由两部分组成：Say（大语言模型给出的方案）和 Can （基于强化学习的打分，给出feasible的行动方案）

### 输入输出

**输入：**

- **自然语言指令：** 用户给出的描述任务的自然语言指令，可以是长期、抽象或含糊的。
- **技能集合（Skills）：** 机器人具备的一系列基本行为技能，如“拿起海绵”或“去桌子那里”。
- **环境状态：** 机器人当前所处的环境状态，包括物体的位置、机器人的状态等。

**输出：**

- **行动序列：** 机器人根据输入的自然语言指令和当前环境状态，通过模型处理后，输出一系列行动步骤，指导机器人完成任务。

### 模型结构

**SayCan** 模型结合了大型语言模型（LLMs）和预训练的技能（skills），通过以下组件实现：

1. **大型语言模型（LLM）：** 用于理解高层次的指令，并生成可能的行动步骤的文本描述。这些模型能够提供任务的语义知识，但缺乏对物理世界的具体理解。
2. **预训练技能（Skills）：** 机器人具备的一系列基本行为，如抓取、移动、放置物体等。每个技能都有一个与之关联的价值函数（affordance function），该函数评估在当前状态下执行该技能的可能性。
3. **价值函数（Value Functions）：** 通过强化学习（RL）训练得到，用于评估在特定状态下执行特定技能的成功率。这些函数为语言模型提供了“现实世界”的锚定，使其能够根据机器人的实际能力选择可行的行动步骤。
4. **行动选择机制：** 结合语言模型生成的技能描述的概率和价值函数评估的执行概率，选择最有可能成功并推进任务完成的技能。
5. **迭代执行与更新：** 机器人执行选定的技能，然后根据执行后的环境状态更新语言模型的输入，继续选择下一个技能，直到任务完成。

我感觉比较玄乎的是强化学习的部分。先补一下强化学习然后回来看看吧。

## LLM + Robotics

肯定会存在的一个方向，被吹成“最接近通用人工智能的”LLM，肯定是一个可行的技术路线，但是上限我觉得不好说。

Awesome 仓库：

[GT-RIPL/Awesome-LLM-Robotics: A comprehensive list of papers using large language/multi-modal models for Robotics/RL, including papers, codes, and related websites (github.com)](https://github.com/GT-RIPL/Awesome-LLM-Robotics)

## PaLM-E: An Embodied Multimodal Language Model

究极多模态缝合，

“论文提出了一个具身多模态语言模型，通过将真实世界的连续传感器模态直接融入语言模型中，实现了单词和感知之间的联系。实验结果表明，PaLM-E可以处理来自不同观察模态的各种具身推理任务，并在多个实现上表现出良好的效果。最大的PaLM-E-562B模型拥有562亿个参数，除了在机器人任务上进行训练外，还是一个视觉语言通才，并在OK-VQA任务上取得了最先进的性能。”

实际上我觉得他这个在控制方面输出还是比较模糊的。

OK-VQA https://okvqa.allenai.org/ 是视觉语言的数据集，和 robotics 无关。

论文里对输出的描述是：

PaLM-E is a generative model producing text based on multi-model sentences as input. In order to connect the output of the model to an embodiment, we distinguish two cases. 

在多模态问询方面，If the task can be accomplished by outputting text only as, e.g., in embodied question answering or scene description tasks, then the output of the model is directly considered to be the solution for the task.

在具身方面，Alternatively, if PaLM-E is used to solve an embodied planning or control task, it generates text that conditions lowlevel commands. In particular, we assume to have access to policies that **can perform low-level skills from some (small) vocabulary**, and a successful plan from PaLM-E must consist of a sequence of such skills.

实际上输出的仍然是 low-level 的自然语言指令。

## VoxPoser: Composable 3D Value Maps

重量级，李飞飞那边的工作。

解决的是 grounding 的问题，也就是如何把 LLM 的指令转化为实际动作。我觉得还是很重要的，感觉很多工作最后的输出也就是自然语言，没有把 action 落到实处。

- **核心思想**：利用LLMs推断出语言条件下的可供性（affordances）和约束，并通过编写代码与VLMs交互，生成3D价值地图（value maps），将这些知识转化为机器人可感知的空间中的实体。
- **方法**：通过生成Python代码，调用视觉API（如CLIP或开放词汇检测器）获取相关对象的空间几何信息，然后操作3D体素来在观察空间中指定奖励或成本。
- **应用**：这些生成的价值地图作为运动规划器的目标函数，直接合成机器人轨迹，实现给定指令，无需额外的训练数据。

### 模型结构：

1. **大型语言模型（LLM）**：
   - 负责理解自然语言指令，并推断出任务相关的可供性（affordances）和约束。
   - 利用其代码编写能力，生成Python代码以调用视觉API和执行数组操作。
2. **视觉-语言模型（VLM）**：
   - 负责处理视觉输入，如通过开放词汇检测器识别物体和获取空间几何信息。
3. **3D价值地图**：
   - 由LLM生成的代码与VLM的输出相结合，形成3D价值地图，这些地图在机器人的观察空间中定义了任务的目标和约束。
4. **运动规划器**：
   - 使用3D价值地图作为目标函数，通过零阶优化方法（如随机采样轨迹）合成机器人轨迹。
5. **动态模型**：
   - 可选组件，用于在线学习环境的动态特性，特别是在涉及复杂接触交互的任务中。

### 输入：

1. **自然语言指令**：
   - 用户给出的任务描述，如“打开顶部的抽屉并注意那个花瓶”。
2. **RGB-D观察**：
   - 机器人当前的三维环境观察，通常包括颜色和深度信息。

### 输出：

1. **3D价值地图**：
   - 包括：
     - **可供性地图**：指导机器人动作的目标位置。
     - **约束地图**：定义了应避免的区域，如附近的花瓶。
     - **旋转地图**：指定末端执行器的朝向。
     - **速度地图**：指定动作的速度。
     - **夹持器地图**：控制夹持器的开闭状态。
2. **机器人轨迹**：
   - 一系列6自由度末端执行器的关键点，用于执行任务。

### 工作流程：

1. **指令解析**：
   - LLM解析自然语言指令，推断出任务的子步骤。
2. **价值地图合成**：
   - LLM生成代码，调用VLM获取视觉信息，并操作3D数组生成价值地图。
3. **轨迹规划**：
   - 运动规划器使用价值地图作为目标函数，规划机器人的轨迹。
4. **执行与反馈**：
   - 机器人执行规划的轨迹，并通过感知模块反馈执行结果，用于动态模型的学习。