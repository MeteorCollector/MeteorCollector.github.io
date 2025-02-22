---
layout: post
title:  "学习一下最近的 resoning models"
date:   2025-02-22 00:14:00 +0800
categories: posts
tag: llm
---

## References

### kimi的方法

[Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)

[大道至简的o1](https://zhuanlan.zhihu.com/p/19838650037)

[一文揭秘 K1.5](https://zhuanlan.zhihu.com/p/19946557325)

[两万字长文深度解密DeepSeek-R1、Kimi 1.5，强推理模型凭什么火出圈？ ](https://mp.weixin.qq.com/s/W7X5_9uVTstInev3-UjExg)

### deepseek-r1

[DeepSeek R1 论文解读&关键技术点梳理 ](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247489094&idx=1&sn=1ae0a770c4586c7cf3601690d8ef01f5&scene=21#wechat_redirect)

[大道至简的o1](https://zhuanlan.zhihu.com/p/19838650037)

[两万字长文深度解密DeepSeek-R1、Kimi 1.5，强推理模型凭什么火出圈？ ](https://mp.weixin.qq.com/s/W7X5_9uVTstInev3-UjExg)

### 隐表示的方法

[超越思维链？深度循环隐式推理引爆AI圈，LLM扩展有了新维度](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650954486&idx=1&sn=a77a499459a017b8000628c416ab0af3&chksm=8591142de76feb78fed93f1ba02a92c943f74d1e749d167493c47c7a57fcdae2da016007a869&mpshare=1&scene=23&srcid=0212LJaRDLLfQ6JIYQlRocsI&sharer_shareinfo=54ff790c9f1ab6b73bed9dde388248a4&sharer_shareinfo_first=54ff790c9f1ab6b73bed9dde388248a4#rd)

### 低成本ds复刻

[200多行代码，超低成本复现DeepSeek R1「Aha Moment」！复旦大学开源 ](https://mp.weixin.qq.com/s/hFArGyWTRTkQIMeStg279w)

[4500美元验证强化学习「魔力」，1.5B模型也能超越o1预览版，模型、数据、代码全开源 ](https://mp.weixin.qq.com/s/f-0f_4zM3xJkq0goWKOqsg)

[s1: Simple test-time scaling](https://arxiv.org/html/2501.19393v2)

### 其他方法

[CoT-Valve：Long Reasoning CoT 的长度压缩](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247489161&idx=1&sn=8366f8bb03fdf3c69026c8de23eb4a08&chksm=c364d1ccf41358da5aeeb4e6961a36dad20fb25ece8acb36961255f5ef05e088d6a30fa6df82&cur_album_id=3098338677899952131&scene=189#wechat_redirect)

minimax?

### Kimi 1.5 & Deepseek

引用“大道至简”那篇文章的通俗描述，Deepseek 和 Kimi 的训练方式分别为：

Deepseek: sft 和 RL 交替进行，每个阶段管好自己的事情，即 rl 学推理，sft 增强可读性。

    训 zero 的 step1：全程无标注数据的参与，就是认准了一个目标：让模型的 reward 变高。这个阶段别和我谈模型格式错误逻辑混乱这种细节，我不看模型表现，只看 reward。只不过 reward 变高的过程中，发现模型的输出越来越长了，反思能力也自己涌现出来了；基于 zero 训 R1 的 step2：就像是我们做普通的 post training 了，sft没有被抛弃，除了rule-based reward，reward_model 也被请回来了，reject sampling 也出手了。

Kimi: 两手抓，既要学会推理也要确保可读性。

    学霸 K 的想法：我还是一步到位吧，在 step1 学推理的过程中，要时刻监控着模型的说话能力是否还正常。为了达到此目标，模型的输出长度，模型对每一个 prompt 的回答准确率等信息，全程都被严格监控。

> Kimi K1.5 其实更多是从 In-context RL 出发是希望模型去模拟 planning 的过程，而不是去显式的进行 planning，其中就是将 state 和价值等信息都视为一个 language tokens；而 DeepSeek R1 是从纯强化学习的角度出发，通过大规模的强化学习和 rule-based reward 来激活模型的能力，其中核心的观念都是不管模型中间做错了什么，它只要不是重复的 pattern，只要模型最后做对了，我们就认为这是一个好的探索，它是值得鼓励的；反之如果模型一顿探索最后做错了，那么再努力也是错，这是需要去进行惩罚的。

[reference](https://mp.weixin.qq.com/s/W7X5_9uVTstInev3-UjExg)

### MCTS & PRM

将输出的答案中的 token 分句或者以几个 token 为单位作为节点，然后对于解空间进行搜索。这本质上是给了模型一个结构化（树状）的先验。prm很难判断当下的某一步是否是对的（你没法未卜先知地对每一个模型可能采取的步骤进行高标准判断，比如实现标注过了），比较取巧的还是生成 n 步，取 top 来给模型下一步的推理方向提建议。

### Kimi 1.5

在冷启动阶段，we employ prompt engineering to construct a small yet high-quality long-CoT warmup
dataset, containing accurately verified reasoning paths for both text and image inputs，准备了一些具有长 cot 的数据，通过 lightweight 的 sft 让模型初步具有思维链的能力（原文中将能力分解为 planning, evaluation, reflection and exploration），然后再去做 RL。

#### 数学部分

论文里面给出的RL优化目标是这样的：

`$$\max_\theta \mathbb{E}_{(x, y^*) \sim D, (y, z) \sim \pi_\theta} [r(x, y, y^*)]$$`

其中 $\theta$ 是模型的参数，即需要找到让这个目标最大的 $\theta$。$(x, y^*) \sim D, (y, z)$ 中，$x$ 是问题，$y^*$ 是对应的正确答案。$D$ 是训练数据集，表示 $x$ 和 $y^∗$ 是从数据集 $D$ 中采样的。`$(y, z) \sim \pi_\theta$` 中，$y$ 是模型生成的最终答案，$z$ 是模型生成的思考路径（COT, Chain of Thought）。这些是根据当前策略模型 `$\pi_\theta$`​ 生成的。具体来说，模型根据当前参数 $\theta$ 生成 $y$ 和 $z$。$r(x, y, y^*)$ 是奖励函数（Reward Function），表示模型生成的答案 $y$ 对于问题 $x$ 的正确性，其中 $y^*$ 代表 gt。实际上这么看的话，这个奖励只和最后的结果有关，涉及到的这么多参数只是在告诉你，这是一个有 COT 的模型。

#### 在线镜像下降 Online Mirror Descent

在线镜像下降（Online Mirror Descent，OMD）是一种常用的优化算法，特别适用于求解大规模问题或需要处理大量数据流的情况。它属于梯度下降类算法的一种变体，广泛应用于在线学习、机器学习和凸优化等领域。
主要特点：

1. **在线学习**：与传统的批量学习不同，在线学习（online learning）指的是模型在每次获取新数据时就更新，而不是等待全部数据都准备好后一次性进行优化。在线镜像下降正是为了应对这种数据逐步到达的场景。

2. **镜像法（Mirror Descent）**：镜像法是一种基于对偶（dual）空间的优化算法，它通过引入一个特定的距离度量（通常是一个凸函数，如Kullback-Leibler散度或欧几里得距离）来引导梯度更新。镜像法的核心思想是利用对偶空间的结构，代替传统的梯度下降在原空间中的更新方式。

3. **在线更新**：在线镜像下降每次从数据流中获取一个样本或一个小批量的数据，并基于该数据更新参数。随着每次新数据到来，模型参数会逐步调整，从而实现学习。

在标准的镜像下降中，假设我们有一个优化目标：

`$$\min_{\theta \in \Theta} f(\theta)$$`

其中 $\Theta$ 是参数空间，$f(\theta)$ 是我们要最小化的目标函数。镜像下降更新规则可以写为：

`$$\theta_{t+1} = \arg \min_{\theta \in \Theta} \left( \langle \nabla f(\theta_t), \theta - \theta_t \rangle + \frac{1}{\eta_t} D_{\Phi}(\theta, \theta_t) \right)$$`

`$\langle \nabla f(\theta_t), \theta - \theta_t \rangle$`


其中，`$\nabla f(\theta_t)$` 是当前参数的梯度，`$D_{\Phi}(\theta, \theta_t)$` 是根据某种距离度量（如Kullback-Leibler散度）计算的距离，`$\eta_t$` 是学习率，决定每一步的更新步长。

Kimi 论文里使用的是原版变体。At the i-th iteration, we use the current model `$\pi_{\theta_i}$` as a reference
model and optimize the following relative entropy regularized policy optimization problem. 可以发现这里的公式写的是对偶形式：

`$$\max_\theta \mathbb{E}_{(x, y^*) \sim D} \left[ \mathbb{E}_{(y, z) \sim \pi_\theta} [r(x, y, y^*)] - \tau \text{KL}(\pi_\theta(x) || \pi_{\theta_i}(x)) \right]$$`

(这个 KL 散度 reference 当前状态的模型，构成一个惩罚项，避免过度优化？)

#### 长度惩罚 Length Penalty

对于一个问题生成很多回答，其中最长回答的长度是 max\_len，最短回答的长度是 min\_len。如果最长回答和最短回答的长度相同，那么 reward 为 0

`$$\text{len\_reward}(i) = \begin{cases} \lambda & \text{if } r(x, y_i, y^*) = 1 \\ \min(0, \lambda) & \text{if } r(x, y_i, y^*) = 0 \end{cases}$$`

其中 `$\lambda = 0.5 - \frac{\text{len}(i) - \text{min\_len}}{\text{max\_len} - \text{min\_len}}$`

#### 算法tricks

- **长上下文扩展(Long context scaling)**：拓展上下文窗口得到更好的长推理能力

- **Long2short** 不想要太长的链，于是和一个短上下文模型进行合并。有这么几种合并方法：模型合并、最短拒绝采样、DPO以及长到短强化学习。感觉类似 deepseek r1 里面提到的用大模型蒸馏小模型。
  
  - 模型合并：简单平均它们的权重来合并一个长上下文模型和一个较短的模型，以获得一个新的无需训练的模型。这个方法似乎泛化效果比较好。

  - 最短拒绝采样：对于同一个问题，模型生成的响应长度变化很大，所以可以对同一问题进行n次采样（在的实验中，n=8），并选择最短的正确响应。

  - DPO：利用长上下文模型生成多个响应样本。最短的正确解决方案被选为正样本，而较长的响应则被当作负样本，包括错误的长响应和正确的较长响应（比选定的正样本长1.5倍）。这些正负对形成了用于DPO训练的成对偏好数据。

  - 长到短强化学习，这个策略在标准的强化学习训练阶段之后，选择一个在性能和标记效率之间提供最佳平衡的模型作为基础模型，并进行单独的长到短强化学习训练阶段，这里主要用到一个应用长度惩罚的方案，以进一步惩罚超出期望长度的回应，同时可能进行纠正。

#### 数据：采样策略

- **Curriculum Sampling** We start by training on easier tasks and gradually progress to more challenging ones. Since the initial RL model has limited performance, spending a restricted computation budget on very hard problems often yields few correct samples, resulting in lower training efficiency. Meanwhile, our collected data naturally includes grade and difficulty labels, making difficulty-based sampling an intuitive and effective way to improve training efficiency.  我们从较简单的任务开始训练，逐渐过渡到更具挑战性的任务。由于初始的强化学习（RL）模型性能有限，在非常困难的问题上投入有限的计算资源往往会产生很少的正确样本，从而导致训练效率低下。同时，我们收集的数据自然包括等级和难度标签，因此基于难度的采样是一种直观且有效的方式来提高训练效率。
- **Prioritized Sampling** In addition to curriculum sampling, we use a prioritized sampling strategy to focus on problems where the model underperforms. We track the success rates si for each problem $i$ and sample problems proportional to $1 − s_i$, so that problems with lower success rates receive higher sampling probabilities. This directs the model’s efforts toward its weakest areas, leading to faster learning and better overall performance. 我们跟踪每个问题 $i$ 的成功率 $s_i$​，并根据 $1−s_i$​ 进行采样，使得成功率较低的问题获得更高的采样概率。这将模型的努力引导到其最薄弱的领域，从而加速学习并提高整体性能。

#### Training Process

1. 预训练。(1) 视觉-语言预训练(Vision-Language Pretraining)，在此阶段建立强大的语言基础，并逐步进行多模态整合；(2) 冷却阶段(Cooldown)，使用精心挑选的合成数据来巩固能力，特别是针对推理和基于知识的任务；(3) 长序列激活阶段(Long-context activation)，将序列处理扩展到131,072个标记。

2. 微调，使用基础 sft。50万个示例用于一般问答，20万个用于编程，20万个用于数学和科学，5000个用于创意写作，2万个用于长上下文任务，如摘要、文档问答、翻译和写作。此外，我们构建了100万个文本-视觉示例，涵盖各种类别，包括图表解释、OCR、图像基础对话、视觉编程、视觉推理和带有视觉辅助的数学/科学问题。

3. 大规模 RL 训练系统和 Megatron + vLLM 的混合部署框架

#### 疑点

看了这么多也不知道是怎么让模型的推理长度主动变长的？Long context scaling 部分的原文是 As training progresses, we observe a concurrent increase in both response length and performance accuracy. Notably, more challenging benchmarks exhibit a steeper increase in response length, suggesting that the model learns to generate more elaborate solutions for complex problem. 这里面主动调高的只有上下文窗口而已。一开始确实用了有链（有plan）的数据来 sft，形成一个初步的风格，但是这个长度主要是 llm 自己为了提高困难问题的准确性学着变长的？wtf？

### DeepSeek R1

#### R1 Zero

纯 RL，奖励分两个，一种是准确率奖励，只和最后答案的准确率有关；第二个是格式奖励，这个在我看来是逼迫着模型思考，利用 thinking token 将思考的过程圈起来。没有 Reward Model，用规则来打分（防止 model 被 hack）。用 GPRO 做强化学习算法。之后就是熟悉的故事了，ds学会了自己反思，并有了 aha moment 的涌现。

有一说一，我觉得是 V3 这个 671B 的模型够大才能这么搞，否则按比较小的模型的能力，不太可能涌现出思考过程。

#### 从 R1 Zero 到 R1

使用 671B v3 作为基座，冷启动时使用从人类和 R1 Zero 那边来的高质量带链的思维数据，让模型获得基本能力。原话是 finetune。

接着再用大规模 RL，仍然有准确率奖励和格式奖励。增加语言一致性奖励确保可读性。

然后为了可读性和非 reasoning 的能力再去 sft

再多领域 RL （Reinforcement Learning for all Scenarios）

#### 结果监督

可以发现 deepseek-r1 和 kimi 都绕过了过程 reward，用结果上的判断倒逼思维链的涌现。不过在冷启动阶段，仍然需要高质量的 cot。

### latent space 方法

在架构上下文章，在输出 token 之前多迭代。感觉和 post training 关系不大。它并没有引导模型做显式的文字上的推理，而是在 hidden state 里手动增加推理时计算的 scale，他加了循环层来让 transformer 在处理完一个 token 之前可以迭代任意次。通过增加计算量的方法促进模型“多思考”。模块名称起得十分诗意，Prelude（前奏）负责将输入编码进 latent space，Core Recurrent Block负责循环，Coda（结尾）负责将 latent space 里面的信息解码出来。

原文：

> Compared to a more stan-
dard approach of long context reasoning (OpenAI, 2024;
DeepSeek-AI et al., 2025), latent recurrent thinking has sev-
eral advantages.

> Latent reasoning does not require construction of bespoke
training data. Chain-of-thought reasoning requires the
model to be trained on long demonstrations that are con-
structed in the domain of interest. In contrast, our pro-
posed latent reasoning models can train with a variable
compute budget, using standard training data with no spe-
cialized demonstrations, and enhance their abilities at test-
time if given additional compute.

> Latent reasoning models require less memory for train-
ing and inference than chain-of-thought reasoning mod-
els. Because the latter require extremely long context
windows, specialized training methods such as token-
parallelization (Liu et al., 2023a) may be needed.

也就是说它是不需要 cot 数据集的，普通训练就好。

但是到底要循环多少次最好，谁也不知道。在训练的时候，论文里是随机采样循环次数的，训练的结果就是可以适应不同深度，也就是不同循环次数的任务。在后面的分析，作者指出模型可以根据问题的难易程度来动态决定循环的深度，指标是两次迭代之间的 KL 散度小雨一个阈值之前循环的次数。如果两次的参数更新太少，视为思考停止，则 exit。

### s1 - simple test time scaling

不得不说这论文写得非常清楚易懂，我想关心的问题一下子就能找到，不像别的论文还要到处找。

s1 是一个比较神奇的低成本实现，不过值得注意的是他的 base model 是 Qwen32B，虽然训练用的资源少，但是模型规模并不小。只进行了简单sft，使用的数据是 1000 条高质量 cot，在[最新的版本](https://huggingface.co/datasets/simplescaling/s1K-1.1)里面用的是 gemini 和 deepseek 的思考过程 + 结果。论文里的选择标准是兼顾 Diffuculty Quality Diversity。

他的 trick 在 inference 时，使用的方法我直接贴原文：

#### Method

We classify test-time scaling methods into 1) Sequential, where later computations depend on earlier ones (e.g., a long reasoning trace), and 2) Parallel, where computations run independently (e.g., majority voting) (Snell et al., 2024; Brown et al., 2024). We focus on sequential scaling as intuitively we believe it should scale better, since later computations can build on intermediate results, allowing for deeper reasoning and iterative refinement. We propose new sequential scaling methods and ways to benchmark them.

#### Budget forcing

We propose a simple decoding-time intervention by forcing a maximum and/or minimum number of thinking tokens at test time. Specifically, we enforce a maximum token count by simply appending the end-of-thinking token delimiter and “Final Answer:” to early exit the thinking stage and make the model provide its current best answer. To enforce a minimum, we suppress the generation of the end-of-thinking token delimiter and optionally append the string “Wait” to the model’s current reasoning trace to encourage the model to reflect on its current generation. Figure 3 contains an example of how this simple approach can lead the model to arrive at a better answer.


#### Baselines

We benchmark budget forcing with: (I) Conditional length-control methods, which rely on telling the model in the prompt how long it should generate for. We group them by granularity into (a) Token-conditional control: We specify an upper bound of thinking tokens in the prompt; (b) Step-conditional control: We specify an upper bound of thinking steps, where each step is around 100 tokens; (c) Class-conditional control: We write two generic prompts that tell the model to either think for a short or long amount of time (see §D.1 for details). (II) Rejection sampling, which samples until a generation fits a predetermined compute budget. This oracle captures the posterior over responses conditioned on its length.

也就是说在 inference 的时候强制让模型多说些内容，说着说着就会反思了。

怎么说呢，这并不是一个“训练出reasoning model”的方法，而是在推理的时候激发出 reasoning 能力的 trick。不过这是不是说明“只要数据够好，sft就能涨点？”

[这里还有一篇知乎可以看看](https://zhuanlan.zhihu.com/p/21602993558)

### Simple GRPO

只是 deepseek-r1 的 GRPO 的简单实现而已，没有什么新东西，可以 play 一下。

[github](https://github.com/lsdefine/simple_GRPO)

### DeepScaleR

[原版 blog](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)

Base model 是 Deepseek-R1-Distilled-Qwen-1.5B，说是把原版练 1.5B 需要的 A100 卡时从 70K 减到了 3.8K。在 1.5B 小模型上做 reasoning，trick 主要有：

逐步增大上下文窗口，8K -> 16k -> 24K

然后就......没了？感觉这个工作的意义还是复现，并证明了 ds 的思路在小模型上是有效的。

#### COT Valve

这个是缩 cot 的工作，用了一个由长到短cot组成的数据集来实现渐进式缩短。这个就不多写了。