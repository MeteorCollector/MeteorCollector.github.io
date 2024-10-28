---
layout: post
title:  "VLM测评，随便写写"
date:   2024-10-25 10:00:00 +0800
categories: posts
tag: util
---

## 写在前面

出于最近忙的项目的考虑，来测评或者了解一下现在市面上的 vision-language models

## OpenAI o1

是否开源：否

需要资源：调用api，需要钱。据说很贵：Developer access to o1 is *really* expensive: In the API, o1-preview is \$15 per 1 million input tokens, or chunks of text parsed by the model, and \$60 per 1 million output tokens. For comparison, GPT-4o costs ​\$5 per 1 million input tokens and ​\$15 per 1 million output tokens.

运算速度：几秒一个 query？或者更长：The model "thinks" for around 10 seconds before starting to write its answers. In our tests, some tasks have taken the model more than a minute of "thought time" before answering. Then, add another bit of time for the model to write a huge and long chain of thought process before giving you the simplest answer in the world. Not good, if patience isn’t your strongest suit. (reference: [OpenAI's o1: The Good, the Bad, and the Ugly of AI's Latest Brainchild - Decrypt](https://decrypt.co/249735/openais-o1-review-good-bad-ugly-ai-latest-brainchild))

### 模型效果

官方的 review: [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)

oi被认为是 “the first reasoning model that shines in really hard tasks”，它的最大进步就是 reasoning 方面的进步。

The training behind o1 is fundamentally different from its predecessors, OpenAI’s research lead, Jerry Tworek, tells me, though the company is being vague about the exact details. He says o1 “has been trained using a completely new optimization algorithm and a new training dataset specifically tailored for it.”    ([reference: OpenAI releases new o1 reasoning model - The Verge](https://www.theverge.com/2024/9/12/24242439/openai-o1-model-reasoning-strawberry-chatgpt))

OpenAI taught previous GPT models to mimic patterns from its training data. With o1, it trained the model to solve problems on its own using a technique known as reinforcement learning, which teaches the system through rewards and penalties. It then uses a “chain of thought” to process queries, similarly to how humans process problems by going through them step-by-step. ([reference: OpenAI releases new o1 reasoning model - The Verge](https://www.theverge.com/2024/9/12/24242439/openai-o1-model-reasoning-strawberry-chatgpt))

为了提高模型的逻辑推理能力，而不是简单地最大似然模仿 pattern，ChatGPT 也开始搞 chain of thought 这一套了。**甚至在交互的时候，模型也会显示它一步一步推理的过程，试图更还原人的思考方式**。由于是多步推理，o1 的开销骤然变大也是情有可原的了。同时也搞了 RL 的 reward system，这就更复杂了。

在官方文档里提到 chain of thought 的验证使用的是 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 。这篇文章里涉及到了很多 train 和 verify 多步推理模型的方法，主要分成两个流派，Outcome-supervised reward models (ORMs) are trained using only the final result of the model’s chain-of-thought, while process-supervised reward models (PRMs) receive feedback for each step in the chain-of-thought. 也就是结果监督（outcome supervision）过程监督（process supervision）两种方法，结果监督仅提供最终结果的反馈，而过程监督则为每一步推理提供反馈。文章作者的验证表明，process supervision can train much more reliable reward models than outcome supervision，而过程监督可以通过 RL 的奖励框架进行。鉴于这篇文章也是 openAI 的人写的，有理由怀疑 o1 的技术路线和这篇文章里讲的也大差不差。后面会详细记一下这篇文章

“The model is definitely better at solving the AP math test than I am, and I was a math minor in college,” OpenAI’s chief research officer, Bob McGrew, tells me. He says OpenAI also tested o1 against a qualifying exam for the International Mathematics Olympiad, and while GPT-4o only correctly solved only 13 percent of problems, o1 scored 83 percent.

In online programming contests known as Codeforces competitions, this new model reached the 89th percentile of participants, and OpenAI claims the next update of this model will perform “similarly to PhD students on challenging benchmark tasks in physics, chemistry and biology.”

At the same time, o1 is not as capable as GPT-4o in a lot of areas. It doesn’t do as well on factual knowledge about the world. It also doesn’t have the ability to browse the web or process files and images. Still, the company believes it represents a brand-new class of capabilities. It was named o1 to indicate “resetting the counter back to 1.” ([reference: OpenAI releases new o1 reasoning model - The Verge](https://www.theverge.com/2024/9/12/24242439/openai-o1-model-reasoning-strawberry-chatgpt))

可以说 o1 在逻辑推理方面特化了，其他方面的推理能力和它的前驱模型相比并没有明显的进步。毕竟这些用不上 chain of thought。但是 doesn't have the ability to browse web or process files and images 是什么意思？它并不能做多模态任务？

### Let's Verify Step by Step

中文资料：[OpenAI最新研究Let's verify step-by-step，过程胜于结果！](https://mp.weixin.qq.com/s/bvrJKy8dufRF0KfC90PDMA)    [OpenAI Let’s Verify Step by Step详细解读 - 知乎](https://zhuanlan.zhihu.com/p/635335926)

这里别人写得很好就不自己写了，有些浪费时间，搬运一下吧。

#### Methods

1. 实验步骤和方法：

2. 1. **训练最可靠的reward model**：对GPT-4模型进行微调，拿到最可靠的ORM和PRM（基于给出的答案和正确的答案的相似度进行打分）。
   2. **生成器**：通过GPT-4生成所有候选解决方法，此步GPT-4没经过RL来alignment优化。
   3. **评价**：对生成的结果进行N选1，最终根据答案来评分。
   4. **两种不同规模的模型**：所有大模型是通过GPT-4微调，没有经过RL训练，小规模模型和GPT4类似，但是计算量少200倍，模型在15亿数学相关的数据集MathMix上进行了微调。
   
3. 过程反馈数据收集方法：

	- **数据收集方案【基础方案】**：对于每一步收集人类反馈结果
	- **优化策略【高价值负样本挖掘】**：标注数据的时候，尽可能对更有可能欺骗reward模型的数据来进行标注，如果展示明显错误的解决方案，获得的反馈价值没那么大
	- **迭代训练奖励模型【高价值负样本挖掘】**：在每次迭代中，对每个问题生成N个解决方案，并仅向数据标注者展示得分最高的K个具有说服力的错误答案解决方案。作者尝试将此top-K过滤应用于问题级别（每个问题K个解决方案）或全局级别（总共K个解决方案，在问题之间不均匀分布）

1. ORM以及PRM建模方法

	1. Outcome-supervised Reward Models (ORMs)：直接判断一个solution最终结果是正确还是错误的【有可能中间推理错误，最终结果正确的现象】。
	2. Process-supervised Reward Models (PRMs)：加入了每一步step的标记，这样可以直接在自回归模型进行训练，同时在遇到结束位置标记时，训练PRMs去预测每一step是否正确。
	3. 如何解决ORM和PRM监督信号不对等的问题：在提供过程监督时，他们有意选择只监督到第一个错误的步骤。这样做使得结果监督和过程监督之间的比较更加简单明了。对于正确的解决方案，两种方法提供相同的信息，即每个步骤都是正确的。对于不正确的解决方案，两种方法都揭示了至少存在一个错误，而过程监督还揭示了该错误的具体位置。如果他们在第一个错误之后提供额外的过程监督，那么过程监督将具有更大的信息优势。这个决策还保持了对人类的标注成本相似：在不依赖于易于检查的最终答案的情况下，确定解决方案的正确性等价于确定其第一个错误。
	
#### 模仿者和先驱们

在 o1 公布后，大家自然是争相复现，比如 g1 模型 [https://github.com/bklieger-groq/g1](https://github.com/bklieger-groq/g1)，但是 thought-chain 实在不是一个新概念。DriveLM 就有比较 fixed 的 thought-chain。

## BLIP-3-Video

是否开源：是

需要资源：没看到最低配置

运算速度：H100 (80GB显存)  1024 token: 3.3 "samples / s" , 推荐的 32 token (毕竟论文题目是 you only need 32 tokens)：8.2 "samples / s"

模型效果：

[论文链接]([xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs](https://arxiv.org/html/2410.16267v1))

关于速度的细节在 Speed 那一节。

We measure the runtime of our models in the training setting for the fair comparison. Here, we report ‘samples per second per GPU’. Without the temporal encoder (i.e., directly using 1024 visual tokens), the model processed 3.3 samples per second. With 16/32/128 tokens using the temporal encoder, the model was able to process 8.5 / 8.2 / 7.5 samples per second.

## 传统 Q-A 对的工作和标注流程

#### DriveLM

对于每个场景中的object，都进行 perception - prediction - planning 的 thought-chain 模式标注。Q 是有模板的。其实他这个也不是 chain，是 graph，多个一级结论决定多个二级结论，然后二级结论生成三级结论......

#### DRAMA

有 perception 和 planning 的 chain-of-thought，不过不是很严谨？

#### Rank2Tell

what - which - where - how - why，但是实际上标注的是 caption，不涉及到自然语言

#### NuScenes-QA

通过模板生成 perception 相关的语句，对于每个 object，描述它们之间的状态以及相互之间的关系

#### NuPrompt

这个生成的是"prompt"，实现的效果是用 prompt 输入，返回描述的 object，实际上也是一个 perception 方向的标注。我看了 dataset ，是一堆json，每一个都含有了符合某个 prompt（例如 persons-who-are-walking）的所有 object，可以通过取交集的方式获取多个prompt描述的object。

#### 总结

除了 DriveLM 有 chain-of-thought，别的dataset还就真的只停留在vqa甚至perception上，搞 full-stack 的还是少

## Thoughts

其实如果从 b2d 延展开来，b2d 本身的创新点一个是闭环，一个是 corner case 的开环。那和 LLM 这些东西一起考虑......

模型扩容：corner case，但是意义不大...... 如果做 corner case 的 dataset，也最好要做 chain-of-thought

验证方式创新：close-loop？但是 AD 的 VLM 要怎么 close-loop ... 对每个状态依赖有 privileged 信息的 teacher model 进行 rule-base 的标定？（后来又想了一下，不可能枚举出各种情况。DriveLM 是提到要做close-loop的，而且他CARLA数据的标注过程是 rule-based 来生成的 q-a 对，之后再人为检查，如果有 privileged info 且有可靠的 rule，是不是真的可以即时生成 q-a 对实现闭环评估？感觉即时生成可能是一个更可靠的 approach？但是这样 video-language model 有点没必要用了。不过planning仍然是比较棘手的，目前能想到的只有同时跑一个有特权信息的teacher model）

注：先自己想的再看的 DriveLM，结果发现我想的他都想到了，自闭

novel的方式：vision -> thought chain？但是不一定每个模型都是这样想的，根据thought chain的每一步问question标定q-a对又不是很novel。不过我觉得确实有一些benchmark的方向可以考虑：

1. 分步，vision -> perception的score，perception -> prediction 这一步的score，prediction -> planning 这一步的 score，甚至还可以再排列组合：perception -> planning，vision -> prediction，vision -> planning （哦不这不就是 e2e 了吗）不过这个有一个问题就是，对于 e2e 的模型，检验非 vision 作为输入的情况并不方便，如果没提供这些接口要怎么输入中间的信息呢？直接输入进模型做 multi-shot 的 LLM 问答吗？那上游的 vision 信息要完全剔除，有些麻烦的。vision -> 各种阶段应该都是可以的。
2. 真的像 o1 一样，对 LLMAD 的逻辑推理过程进行打分。类似 RL 的奖赏规则？但是感觉 AD 的思考过程比较比较 fixed ，不需要太复杂的框架？再细想想，可能 LLMAD 的思考步骤确实不会出问题，大家都是 p -> p -> p，还是更有可能在每一步的推理结果出现问题。这就又回到了上面，分步的这个考量。又变成了上面这个问题。
3. 如果全面向 o1的推理路线靠拢，不把步骤严格限制成 p-p-p 而是按照 o1 的来，那需要解决 vision -> perception （用 vlm）。相当于 o1 已经得知了 perception 的结果，有了特权信息。但是如果用 carla，特权信息不用 vlm  来解决，是不是可以直接获取呢？
4. 如果要练类似 o1 的过程监督型llm，需要过程相关的数据集的话：对于 p-p-p 的每一步用 GPT-4 或者其他模型生成候选方案，然后打分（在 step-by-step 那篇论文里面打分也是用llm来打的），分为正面-负面-有歧义。在论文里还提到尽可能对更有可能欺骗reward模型的数据来进行标注，是不是要对 reward 模型有一定的洞察呢？往后过程监督模型的实现就比较 RL 了，可以作为 benchmark 的一部分但是并不是 dataset 的一部分。数据生成这边就可以像 1 2 这样实施了。
5. 综上所述我觉得可行的大思路是先做一个正确的 chain-of-thought 的 dataset，保证它的结果是正确的。先用 vlm 或者 carla 的特权信息（后者更省事一点）获得 percerption，后续 prediction / planning / behaviour 有几种选择：最传统的是让 teacher model 或者其他可靠的模型开，可以用 rule-base 生成自然语言然后用 gpt 润色。如果让 gpt 等等大语言模型开，o1 会有自己的 chain （当然也可以 specify p-p-p 的结构，每一步通过 rule-based 的方法给出这一步的结论，看 gpt 下一步怎么推理）。针对之前提到的 teacher model 犯错的问题：多数投票或者手动检查？最后如果要模仿 step by step 那个论文，需要负样本和中性样本等等，就在每一步用生成模型生成很多不同的决策，再进行打分。为什么不从一开始就生成所有方案呢？因为要的错误方案的上一步是正确方案才行，要不然有些浪费——基于错误的上一步得到错误的下一步的解空间太大了。而且过程监督奖励模型也是检测到第一个错误就不再检测了。

DriveLM 做过了 thought chain，是 perception -> prediction -> planning 的结构，被描述为 "full stack"

