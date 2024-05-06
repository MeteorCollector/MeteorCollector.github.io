---
layout: post
title:  "论文阅读：ai视觉心理学"
date:   2024-05-06 11:24:00 +0800
categories: posts
tag: psychology
---

## 写在前面

很长时间没有写文章了，主要是这段时间在忙 `multi-shot` 数据集采集和复习 `408` 的工作。唉，南哪的夏令营考核太繁杂了，简直比考研还苦难，本校何苦为难本校呢？

Anyway，现在开始要做一个关于计算机视觉和心理学交叉的项目，有很多论文要看。所以开几个 markdown 做一下论文阅读笔记。

学长在发给我们论文之前已经写好了一个大纲，所以这一篇也从大纲的基础上向外延伸。

# AI视觉心理：相关文献与资料

## 1. 心理理论与应用

**实验平台的理论基础，箱庭测验与视觉投射理论**

**关注：1）箱庭疗法，沙盘的基础理论书籍；2）同理心AI心理治疗，一系列论文的组织逻辑是怎样的？如何将AI算法和心理理论结合起来？3）组内心理相关的工作，初步了解沙盘测评与疗愈**

   - 沙盘与箱庭理论
     - 箱庭疗法 张日昇等
   - 实例应用：同理心AI心理治疗
     - A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support 
     - Towards Facilitating Empathic Conversations in Online Mental Health Support: A Reinforcement Learning Approach
     - Human–AI collaboration enables more empathic conversations in text-based peer-to-peer mental health support 
   - 心理沙盘相关工作
     - 基于证据中心设计理论的智能心理测评：构建与应用
     - AI心世界测评原理与数据效果
     - 基于游戏的心理测评
     - 我的世界论文
     - A Hierarchical Theme Recognition Model for Sandplay Therapy

## 2. 视觉理解与分析

**早年的一些工作，使用Deep Learning算法解决经典视觉任务**

**未使用预训练模型，End-to-End** 

**关注：1）在什么数据集上验证？数据集的难度与挑战如何？2）方法的基本框架是怎样的，创新点在何处？3）算法的效果如何，相比其他方法优势在哪里？**

   - Image Caption
     - Auto-encoding_and_Distilling_Scene_Graphs_for_Image_Captioning
     - Dense_Captioning_with_Joint_Inference_and_Visual_Context
     - Dense_Relational_Captioning_Triple-Stream_Networks_for_Relationship-Based_Captioning
     - DenseCap_Fully_Convolutional_Localization_Networks_for_Dense_Captioning
     - Dense-Captioning_Events_in_Videos
   - Visual Question Answering
     - FVQA_Fact-Based_Visual_Question_Answering
     - Out of the Box：Reasoning with Graph Convolution Nets for Factual Visual Question Answering
     - Visual_Genome
     - zs-f-vqa

## 3. 语言模型

**国内外主流的Large Language Model及相关应用**

**关注：1）不同语言模型的性能、综合表现如何？参数量与推理开销？2）语言模型接口设计、API调用、微调**

   - 大语言模型
     - ChatGPT：https://chat.openai.com/
     - 通义千问：[通义 (aliyun.com)](https://tongyi.aliyun.com/qianwen/)
     - 书生：https://github.com/internLM/int
     - 智谱清源：https://chatglm.cn/
   - 实例应用
     - EmoLLM：[SmartFlowAI/EmoLLM: 心理健康大模型、LLM、The Big Model of Mental Health、Finetune、InternLM2、Qwen、ChatGLM、Baichuan、DeepSeek、Mixtral、LLama3 (github.com)](https://github.com/SmartFlowAI/EmoLLM)
     - MindChat：•[X-D-Lab/](https://github.com/X-D-Lab/MindChat)[MindChat](https://github.com/X-D-Lab/MindChat)[: ](https://github.com/X-D-Lab/MindChat)[🐋](https://github.com/X-D-Lab/MindChat)[MindChat](https://github.com/X-D-Lab/MindChat)[（漫谈）](https://github.com/X-D-Lab/MindChat)[——](https://github.com/X-D-Lab/MindChat)[心理大模型：漫谈人生路](https://github.com/X-D-Lab/MindChat)[, ](https://github.com/X-D-Lab/MindChat)[笑对风霜途 ](https://github.com/X-D-Lab/MindChat)[(github.com)](https://github.com/X-D-Lab/MindChat)

## 4. 沙盒模拟与智能体

**利用沙盒环境，构建仿真场景，从而模拟并研究人类/智能体在社会环境中相互作用与进化的一类工作** 

**关注：1）如何构建环境；2）使用了什么方法、模拟了人类的哪些特性；3）采用什么算法，取得怎样的效果**

   - Agent Sims: An Open-Source Sandbox for Large Language Model Evaluation
   - Emergent Social Learning via Multi-agent Reinforcement Learning
   - Humanoid Agents: Platform for Simulating Human-like Generative Agents
   - Social diversity and social preferences in mixed-motive reinforcement learning