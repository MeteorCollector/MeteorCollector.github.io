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
       - 这是一本五百多页的书，里面是很详细的，从箱庭疗法的理论依据、其中元素的象征意义到具体的案例，在此就不细读，可以以后再来参考，当作工具书使用。
   - 实例应用：同理心AI心理治疗
     - **A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support**
       - 这是一篇用机器学习方法识别 “共情” 的文章。提出了一个叫做 ***EPITOME***  *(**E**m**P**athy **I**n **T**ext-based, asynchr**O**nous **ME**ntal health conversations)* 的 “共情” 识别模型。
       - Abstract: 共情对成功的心理健康支持至关重要。共情测量主要发生在同步、面对面的环境中，可能无法转化为异步、基于文本的情境。由于数百万人使用基于文本的平台进行心理健康支持，了解这些情境中的共情至关重要。在这项工作中，我们提出了一种计算方法，以了解共情在在线心理健康平台中的表达方式。我们开发了一个新颖的理论基础的统一框架，用于描述基于文本的对话中的共情沟通。我们收集并分享了一组包含10k个（帖子、回复）配对的语料库，使用这个共情框架进行了注释，并提供了支持注释的证据（理由）。我们开发了一个基于多任务RoBERTa的双编码器模型，用于识别对话中的共情并提取其预测的理由。实验证明，我们的方法能够有效地识别共情对话。我们进一步应用这个模型来分析235k个心理健康互动，并显示用户不会随着时间自学共情，揭示了共情培训和反馈的机会。
       - 在技术上看起来还是有 Transformer 的影子在的，首先将 seeker 的数据 encode，然后将 rationale 的数据 encode，两者合成一个 attention，再将这个 attention 与 encoded rationale 作为输出分别输入 Empathy Identifier 和 Rationale Extractor。这被叫做 “Bi-Encoder Model with Attention” 
     - **Towards Facilitating Empathic Conversations in Online Mental Health Support: A Reinforcement Learning Approach**
       - 这个工作把上一篇论文的 “识别工作” 抬高到了应用层面，把自然语言改写成更加具有同理心的形式。
       - Abstract: 在线点对点支持平台使数百万寻求和提供心理健康支持的人进行对话成为可能。如果成功，基于网络的心理健康对话可以提高治疗的获取性，并减少全球疾病负担。心理学家反复证明，共情，即理解和感受他人情感和经历的能力，是在支持性对话中取得积极结果的关键组成部分。然而，最近的研究表明，在线心理健康平台上高度共情的对话是罕见的。在本文中，我们致力于改善在线心理健康支持对话中的共情。我们引入了一项新任务，即共情改写，旨在将低共情的对话帖子转化为更高的共情水平。学习这样的转换是具有挑战性的，并需要对共情有深入的理解，同时通过文本流畅性和对话上下文的特定性来保持对话质量。在这里，我们提出了 **PARTNER**，一个深度强化学习（RL）代理程序，它学会对帖子进行句子级别的编辑，以增加表达的共情水平，同时保持对话质量。我们的RL代理利用了一个基于转换器语言模型的策略网络，该模型是从GPT-2进行了调整，它执行生成候选共情句子并将这些句子添加到适当位置的双重任务。在训练过程中，我们奖励能够增加帖子中共情水平的转换，同时保持文本流畅性、上下文特定性和多样性。通过自动和人工评估的组合，我们证明了Partner成功地生成了更具共情、具体和多样化的回复，并且优于与样式转换和共情对话生成等相关任务的NLP方法。这项工作对于在基于网络的平台上促进共情对话具有直接的意义。
       - 在技术方面，主要的思路是 “在原来的基础上插入有同理心的句子”。模型使用 Seeker Post 和 Response Post 作为输入，经过多个 Transformer 模块，分别用 Position Classifier 决定插入位置、用 Sentence Generator 生成句子，对 Response Post 进行 rewrite。评价标准由四个评价函数组成：Change in empathy reward ($r_e$)，Text fluency reward ($r_f$)，Sentence coherence reward ($r_c$)，Mutual information reward ($r_m$)。假若改写后的句子不符合标准，会再输入一次来 rewrite，直到结果令人满意。
     - **Human–AI collaboration enables more empathic conversations in text-based peer-to-peer mental health support**
       - 这篇论文和上一篇差别不大，仍然是增加机器生成语句的共情性。不过我似乎没在这篇文章找到任何技术相关的内容，而是人机交互方式？它们并没有对模型进行 evaluate （采用的是上一篇里的 **PARTNER**）而是对不同的采用 ai 的方式产生的效果进行 evaluate，最后发现 “基于AI的建议进行重写（而并不是照抄）”效果最好。
       - Abstract: 人工智能（AI）的进步使得系统能够增强和与人类合作执行简单的机械任务，如安排会议和语法检查文本。然而，这种人工智能与人类的合作对于更复杂的任务，如进行共情对话，存在挑战，因为AI系统在处理复杂的人类情感和这些任务的开放性质方面面临困难。在这里，我们专注于点对点心理健康支持，这是一个共情对成功至关重要的场景，并研究AI如何与人类合作，在文本的在线支持性对话期间促进点对点的共情。我们开发了**HAILEY**，一个AI-循环代理，提供及时反馈，帮助提供支持的参与者（同龄支持者）更具共情地回应那些寻求帮助的人（支持寻求者）。我们在 TalkLife（N = 300），一个大型在线点对点支持平台上，进行了一项非临床随机对照试验，与真实世界的同龄支持者合作。我们展示了我们的人机合作方法导致了同龄人之间对话共情的整体增加19.6％。此外，我们发现，在将自己标识为在提供支持方面遇到困难的同龄支持者的子样本中，共情增加了38.9％。我们系统地分析了人机合作的模式，并发现同龄支持者能够直接和间接地使用AI反馈，而不会过度依赖AI，同时报告了反馈后的自我效能的改善。我们的发现表明，基于反馈驱动的AI-循环写作系统有潜力在开放性、社交性和高风险任务，如共情对话中赋予人类力量。
   - 心理沙盘相关工作
     - 基于证据中心设计理论的智能心理测评：构建与应用
       - **HIST** 模型，中科院的工作。
       - Abstract: 有效测量和评估抑郁、焦虑、强迫等心理健康维度是心理测量学的重要问题，本研究构建了基于证据中心设计理论的智能心理测评方法，旨在结合经典测量理论和人工智能最新技术开展智能心理测评研究。在测评环境方面，智能心理测评方法引入电子沙盘游戏作为测评环境，相较量表测评和视觉投射测验，该环境为被试提供了更自由、真实、开放的表达空间；在分析技术方面，本方法通过人工智能技术实现了对沙盘作品象征性意义的自动识别和心理分层评估模型的构建，从而对被测者的心理健康状况进行综合评估。经实验验证，基于证据中心设计理论的智能心理测评方法在信度和效度方面表现良好，对心理健康障碍的预警作用显著，具备对强迫、抑郁、焦虑三个心理健康维度的测评能力。基于证据中心设计理论的智能心理测评方法为心理健康问题的测评提供了可靠的工具，为人工智能与心理学的跨学科结合提供了有效的研究范式，能有效促进生成式人工智能技术在心理学领域的应用。
       - 模型结构：“心理维度 - 诊断要素 - 沙盘主题”，识别沙盘主题（通过摆放的元素和位置提取特征 “主题”） $\to$ 获得诊断要素（情绪低落、思维迟缓、精力缺乏、消极预期、易敏感......） $\to$ 获取心理维度（抑郁、焦虑、强迫）。
       - 这被称为 “证据中心设计”，从情境中获得指标。
     - AI心世界测评原理与数据效果
       - 是一个产品介绍。
     - 基于游戏的心理测评
       - 是一篇综述论文。
     - 我的世界论文
       - 摘要：本文介绍了在《我的世界》中实施的第一个基于游戏的智力评估，这是一款非常受欢迎的电子游戏，销量超过2亿份。在游戏的三维沉浸式环境中，实现了基于矩阵的模式补全任务（PC）、心理旋转任务（MR）和空间构建任务（SC）。PC旨在衡量归纳推理能力，而MR和SC则是空间能力的衡量指标。我们测试了129名年龄在10至12岁之间的儿童，分别进行了基于《我的世界》的测试和等价的纸笔测试。所有三个量表符合拉斯模型（Rasch model），并且具有中等可靠性。在区分PC和SC之间的区分方面，因子效度良好，但在MR方面没有找到明确的因子。在潜在水平上，使用《我的世界》和传统测试测量的能力之间的相关性很高（r = 0.72），收敛效度良好。子测试级别的相关性在中等范围内。此外，我们发现从游戏环境中收集的行为日志数据对《我的世界》测试的表现具有很高的预测能力，并在较小程度上也预测了传统测试的得分。我们确定了与空间推理能力相关的许多行为特征，证明了分析粒度化行为数据的实用性，除了传统的答题格式之外。总的来说，我们的研究结果表明，《我的世界》是一个适合于基于游戏的智力评估的平台，并鼓励未来的工作探索在纸上或传统计算机测试中不可行的基于游戏的问题解决任务。
       - 这篇论文并不像我预期的那样是类似沙盒测试的物品摆放分析，而是利用mc进行一系列测试，让我觉得有点捞。主要评估方式是与传统心理测试方法做对比。
     - **A Hierarchical Theme Recognition Model for Sandplay Therapy**
       - **HI**erarchical **S**andplay **T**heme recognition
       - Abstract: 沙盘游戏疗法作为心理投射的关键工具，测试者构建一个场景来反映他们内心世界，而精神分析师则审视测试者的心理状态。在这个过程中，识别沙盘游戏图像的主题（即识别内容和情感色调）是促进更高水平分析的关键步骤。与传统的视觉识别不同，后者仅关注基本信息（例如类别、位置、形状等），沙盘游戏主题识别需要考虑图像的整体内容，然后依赖于分层知识结构来完成推理过程。然而，沙盘游戏主题识别的研究受到以下挑战的阻碍：（1）收集高质量且足够的沙盘游戏图像，并配对专家分析以形成科学数据集具有挑战性，因为此任务依赖于专业的沙盘游戏环境。（2）主题是综合且高水平的信息，使得难以直接采用现有的工作来完成此任务。总之，我们从以下几个方面应对了上述挑战：（1）基于对挑战的仔细分析（例如，小规模数据集和复杂信息），我们提出了 **HIST**（分层沙盘游戏主题识别）模型，该模型融合了外部知识来模拟精神分析师的推理过程。（2）以分裂主题（代表性且均匀分布的主题）为例，我们提出了一个名为 $SP^2$（沙盘游戏分裂）的高质量数据集，用于评估我们提出的方法。实验结果表明，与其他基线方法相比，我们的算法表现出更优异的性能，并且消融实验证实了整合外部知识的重要性。我们期待这项工作将有助于沙盘游戏主题识别的研究。相关数据集和代码将持续发布。
       - 结构：输入图片 $\to$ 语义识别 $\to$ 根据外部信息提取特征 (External Knowledge Incorporation Module) $\to$ 沙盘主题识别 (Theme Classification Module)
       - $SP^2$ 数据集，Considering the challenge of gathering high-quality sandplay images paired with theme annotation from psychoanalysts, we take the representative split theme as the example, and construct the SP2 dataset.

## 2. 视觉理解与分析

**早年的一些工作，使用Deep Learning算法解决经典视觉任务**

**未使用预训练模型，End-to-End** 

**关注：1）在什么数据集上验证？数据集的难度与挑战如何？2）方法的基本框架是怎样的，创新点在何处？3）算法的效果如何，相比其他方法优势在哪里？**

   - **Image Caption**
     
     - **Auto-encoding_and_Distilling_Scene_Graphs_for_Image_Captioning**
     
       - end-to-end encoder-decoder模型存在一个问题：当将一张包括未见过的场景输入到网络中时，返回的结果仅仅就是一些显著的object，比如“there is a dog on the floor”，这样的结果与object detection几乎没有区别
     
         认知上的证据表明，基于视觉的语言并非是end-to-end的，而是与高层抽象的符号相关。
     
         例如，对于一张图片，scene abstraction是“helmet-on-human”和"road dirty"，我们则可以生成"a man with a helmet in contryside"通过使用一个常识：country road is dirty，这种推断就是inductive bias。
     
         本文将inductive bias融合到encoder-decoder中来进行image captioning，利用符号推理和端到端多模型特征映射互补，通过scene graph($ \mathcal{G}$)来bridge它们，一个scene graph($ \mathcal{G}$)是一个统一的表示，它连接了以下几个部分
     
         - objects(or entities)
       
     - their attributes
         - their relationships in an image($\mathcal{I}$) or a sentence($\mathcal{S}$)，通过有向边表示
     
         作者提出了Scene Graph Auto-Encoder(SGAE)，作为一个句子重建网络，其过程是 $\mathcal{S}\rightarrow\mathcal{G}\rightarrow\mathcal{D}\rightarrow\mathcal{S}$
         其中 $\mathcal{D}$ 是一个 可训练的字典，用来记录结点特征，$\mathcal{S}\rightarrow\mathcal{G}$ 使用现成的scene graph language parser，$\mathcal{D}\rightarrow\mathcal{S}$ 是一个可训练的RNN decoder，注意 $\mathcal{D}$ 是"juice"——即language inductive bias，在训练SGAE中得到，通过将 $\mathcal{D}$ 共享给encoder-decoder的pipline：$\mathcal{I}\rightarrow\mathcal{G}\rightarrow\mathcal{D}\rightarrow\mathcal{S}$，即可利用语言先验来指导端到端模型，具体的 $\mathcal{I}\rightarrow\mathcal{G}$ 是一个visual scene graph detector，引入multi-modal GCN来进行 $\mathcal{G}\rightarrow\mathcal{D}$ 的过程，来补足detection的不足之处，有趣的是，$\mathcal{D}$ 可以被视作为一个working memory，用来从 $ \mathcal{I}\rightarrow\mathcal{S}$ re-key encoded nodes，以更小的domain gap来得到一个更一般的表达。
     
         - **Contrubution**
         - 一个先进的SGAE模型，可以学习language inductive bias的特征表达
         
     - 一个multi-model 图卷积网络，用来调节scene graph到视觉表达
         - 一个基于SGAE的 encoder-decoder image captioner with a shared dictionary guiding the language decoding
         
         搬运自 [Auto-Encoding Scene Graphs for Image Captioning 论文阅读笔记-CSDN博客](https://blog.csdn.net/luo3300612/article/details/90042843)
       
     - **Dense_Captioning_with_Joint_Inference_and_Visual_Context**
     
       - Abstract: Dense Captioning 是一种新兴的计算机视觉主题，用于理解具有密集语言描述的图像。其目标是从图像中密集地检测视觉概念（例如对象、对象部分以及它们之间的交互），并为每个概念标注一个简短的描述性短语。我们确定了密集字幕面临的两个关键挑战，在解决该问题时需要妥善处理这两个挑战。首先，每个图像中密集的视觉概念注释与高度重叠的目标区域相关联，使得每个视觉概念的准确定位具有挑战性。其次，大量的视觉概念使得仅通过外观来识别它们变得困难。我们提出了一个基于两个新思想（联合推理和上下文融合）的新模型管道，以缓解这两个挑战。我们以系统的方式设计了模型架构，并对架构的变化进行了彻底评估。我们的最终模型，紧凑高效，实现了在Visual Genome [23]上密集字幕的最先进准确度，相对于先前最佳算法取得了73％的相对增益。定性实验还展示了我们模型在密集字幕中的语义能力。
       - 与传统标注任务的区别是，Dense Captioning 要求尽可能详细地描述目标的特征。
       - 结构：通过卷积分别获得 region proposal、region feature、context feature，region feature 的分割结果 (基于 Faster R-CNN)(获取 RoI) 可以获得一个 detection scrore，由 region feature 和 context feature 获得 caption (Faster R-CNN & LSTM 缝合) 和 bounding box；
       - 验证数据集：Visual Genome dataset (主要), Flicker30k, MS COCO；
     
     - **Dense_Relational_Captioning_Triple-Stream_Networks_for_Relationship-Based_Captioning**
     
       - 提出了 **MTTSNet** 模型 which facilitates POS aware relational captioning，**relational captioning** 方法。
       - 验证数据集：Visual Genome，MS COCO，UCF101 (human interactions with other objects or surroundings)；
       - 所谓 Triple-Stream 是指处理 Union Region、Subject Region、Object Region 的三条 LSTM。
       - 例如“人在骑马”，人和马的整体是 union，人是 subject，马是 object。
       - 模型结构：首先将图片输入 RPN （Region Proposal Network）生成 Region Proposal。接着经过 Combination Layer，每次取一对 Subject 和 Object。Union 区域应该是根据这两个直接计算出来的。接着将这三个区域经过 ROI 池化获得特征图，再过两次全卷积，扔进 Triple-LSTM，最后输入 Multi-task Module。
       - 最后的结果是 POS (subj-pred-obj)；
       - 损失函数：$\mathcal{L} = \mathcal{L}_{cap} + \alpha \mathcal{L}_{POS} + \beta \mathcal{L}_{det} + \gamma \mathcal{L}_{box}$
       - 总损失 = captioning loss + alpha * POS classification loss + beta * detection loss + gamma * bounding box regression loss。其中 alpha bata gamma 是平衡权重。
       - 这篇论文还需要再具体看看。
     
     - **DenseCap_Fully_Convolutional_Localization_Networks_for_Dense_Captioning**
       - 提出了 **DenseCap**，李飞飞他们的工作。
       - Abstract: 我们引入了密集字幕任务，该任务要求计算机视觉系统在自然语言中同时定位并描述图像中的显著区域。密集字幕任务将对象检测泛化为当描述由单个词组成时，以及当一个预测区域覆盖整个图像时的图像字幕生成任务。为了共同解决定位和描述任务，我们提出了一个完全卷积定位网络（FCLN）（Fully Convolutional Localization Network）架构，它通过单一高效的前向传播处理图像，不需要外部区域提议，并且可以通过单轮优化进行端到端训练。该架构由一个卷积网络、一个新颖的密集定位层和一个生成标签序列的递归神经网络语言模型组成。我们在Visual Genome数据集上评估了我们的网络，该数据集包含94,000张图像和4,100,000个区域标注的字幕。我们观察到，与基于当前最先进方法的基线相比，我们的网络在生成和检索设置中都实现了速度和准确性的提高。
       - 有解析文章：[论文笔记-DenseCap - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/91023126)
       - 结构：卷积网络 $\to$ 密集定位层 (dense localization layer) $\to$ RNN
       - 卷积层得到 $C^\prime \times W^\prime \times H^\prime$ 特征图 $\to$ 卷积，获得 region proposal 和 region score, 进行 box regression 得到框体，用双线性插值对得到的特征图尺寸进行变化 $\to$ 接入全连接的 Recognition Network 层 （$D = 4096$）获得 Region  Code $\to$ 用 RNN Language Model 获得基于图片内容的自然语言序列。
       - 感觉这里的重点是 RNN Language Model，剩下的都是 RCNN 之流玩过的东西了，和其他图像识别任务差别不大。但是论文里并没有讲得太详细......还是要再看看。
       - Language Model 的本质是要预测下一个词语。这篇文章就是用这些识别出来的标注信息来输入（原文用的是 condition）自然语言模型。
       - Concretely, given a training sequence of tokens s1, . . . , sT , we feed the RNN T + 2 word vectors x−1, x0, x1, . . . , xT , where x−1 = CNN(I) is the region code encoded with a linear layer and followed by a ReLU non-linearity, x0 corresponds to a special START token, and xt encode each of the tokens st, t = 1, . . . , T . The RNN computes a sequence of hidden states ht and output vectors yt using a recurrence formula ht, yt = f(ht−1, xt) (we use the LSTM [18] recurrence). The vectors yt have size |V |+1 where V is the token vocabulary, and where the additional one is for a special END token. The loss function on the vectors yt is the average cross entropy, where the targets at times t = 0, . . . , T − 1 are the token indices for st+1, and the target at t = T is the END token.
       
     - **Dense-Captioning_Events_in_Videos**
     
         - 也是李飞飞的工作，提出了 **Activity Net**。
         - Abstract: 大多数自然视频包含大量事件。例如，在一个“男子弹钢琴”的视频中，视频可能还包含“另一个男子跳舞”或“人群鼓掌”的场景。我们介绍了密集事件字幕任务，涉及在视频中检测和描述事件。我们提出了一种新模型，能够在单次视频通过中识别所有事件，同时用自然语言描述检测到的事件。我们的模型引入了现有提议模块的变体，旨在捕获持续几分钟的短事件和长事件。为了捕获视频中事件之间的依赖关系，我们的模型引入了一个新的字幕模块，利用过去和未来事件的上下文信息共同描述所有事件。我们还介绍了ActivityNet Captions，一个用于密集事件字幕的大规模基准。ActivityNet Captions 包含20,000个视频，总计849个视频小时，其中包含100,000个总描述，每个描述都有其独特的开始和结束时间。最后，我们报告了我们的模型在密集事件字幕、视频检索和定位方面的性能。
         - 感觉就是把 2D 的工作拓展到 3D 了（工作量升了一维）。
         - 结构：Input Video $\to$ Conv3D $\to$ Video features $\to$ Proposal module $\to$ Proposals $\to$ **Captioning Module** $\to$ Output Captions
         - **Captioning Module**: Proposals $\to$ Representation $\to$ Attention $\to$ Context (引入了past和future) $\to$ LSTM
         - 损失函数：$\mathcal{L} = \lambda_1 \mathcal{L}_{cap}$ （字幕）和 $\lambda_2 \mathcal{L}_{prop}$ （proposal）
         - 感觉目前不太做视频相关的项目，所以先略过了。
     
   - **Visual Question Answering**
     
     - **FVQA_Fact-Based_Visual_Question_Answering**
     
       - Abstract: 视觉问答（VQA）在计算机视觉和自然语言处理社区引起了广泛关注，部分原因是它提供了对两个重要信息源之间关系的洞察。当前的数据集及建立在其上的模型主要关注那些仅通过对问题和图像进行直接分析即可回答的问题。这种不需要外部信息即可回答的问题集合很有意思，但非常有限。例如，它排除了需要常识或基本事实知识来回答的问题。在这里，我们介绍了FVQA（基于事实的VQA），这是一个需要更深层次推理的VQA数据集。FVQA主要包含需要外部信息才能回答的问题。因此，我们通过额外的图像-问题-答案-支持事实元组扩展了传统的视觉问答数据集，每个支持事实都表示为结构三元组，例如<Cat，CapableOf，ClimbingTrees>。我们在FVQA数据集上评估了几种基线模型，并描述了一种新颖的模型，该模型能够根据支持事实对图像进行推理。
       - dataset (previous): DAQUAR, COCO, Visual Gnome, MadLib
       - 这篇文章引入了 Fact，提出了 FVQA 数据集；
       - 建构了知识库 (Knowledge Base, **KB**)，基于 DBpedia，ConcaptNet，WebChild；
       - Question-Query Mapping $\to$ Querying the KB $\to$ answer；
     
     - **Out of the Box：Reasoning with Graph Convolution Nets for Factual Visual Question Answering**
     
       - Abstract: 准确回答关于给定图像的问题需要将观察结果与常识知识相结合。虽然这对人类来说是轻而易举的，但使用常识知识进行推理仍然是一个算法挑战。为了推动这个方向的研究，最近引入了一项新颖的“基于事实”的视觉问答（FVQA）任务，同时提供了一个大量策划好的事实集，这些事实通过一个关系连接了两个实体，即两个可能的答案。给定一个问题-图像对，深度网络技术已被用来逐步减少大量事实，直到最终剩下的事实中的两个实体中的一个被预测为答案。我们观察到，逐步考虑一个事实来形成局部决策的过程是次优的。相反，我们开发了一个实体图，并使用图卷积网络来通过同时考虑所有实体来“推理”正确答案。我们在具有挑战性的FVQA数据集上展示了这一点，与最先进技术相比，这导致了准确率提高约7%。
       - 有解析文章：[NeurIPS 2018亮点选读：深度推理学习中的图网络与关系表征 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/50995189)
       - 提出了基于图卷积的网络试图同步学习事实上下文的推理过程与图像内容理解，之前深度网络筛选事实的这一训练过程用图卷积网络代替它成为一个端到端的推理系统，基于此思想，文章最终取得的准确度比之前的 State of the art 工作高出 7%。
       - 基于事实的图像问答任务（FVQA）输入数据由成对的图像与其对应的问题，外加包含大量事实的知识库（Knowledge Base, KB），图像均为常见的 RGB 彩图，问题则是简单的短句问句，值得一提的是额外的事实（Fact）在知识库中以形如 f=(x,y,r) 的三元组来表示，x 指代图像中的实体信息，y 指代属性或者短语，r 则是关系介词（比如从属关系，是非关系，位置关系等）。
       - 文章提出的方法由两大模块构成：事实追溯（Fact Retrieval）与答案预测（Answer Prediction）。
       - 在第一个模块中，首先得到处理后事实（Fact）、图像中的视觉概念（Visual Concept）与问题（Question）的 Glove 词嵌入（Glove Embeddings）信息，之后计算 Fact 与二者之间的余弦相关性（Cosine Similarity）并按照相关性对追溯到的 Fact 进行排序，同时取正相关的前一百条事实（Top-rank 100 Fact）作为一个初筛的小型知识库，之后根据 LSTM 处理后提取出来的问题中的关系（Relation in Question）作为 Ground-Truth 进一步提取过滤符合该关系下的 Facts，得到最终的相关实体集合 E（Relevant entities under certain relation）
       - 至此第一模块类似于预处理和预筛选的功能全部完成。
       - 进入第二个答案预测（Answer Prediction）的模块，将图像中的视觉概念（Visual Concept）与问题的 Glove 词嵌入表示（Glove Embeddings）以及相关事实的词嵌入表示作为三部分的特征进行连接形成完整的一条大特征实体 e（feature entity）, e 则代表图卷积网络（Graph Convolution Network, GCN）中的一个节点（Node），由两个隐层的 GCN 与一层 MLP 形成的网络进行训练，网络采用随机梯度下降（SGD）与交叉熵损失（Cross-entropy Loss）进行最终的答案预测。
       - 相较于以往的图像问答任务，FVQA 在问题的设计中添加了需要进行关系分析与推理（reason）的部分，在图像理解和自然语言处理方面是一个非常有意义的跨模态任务，而文章采用一种端到端（end-to-end）的形式把涉及到的数据输入做了一个特征拼接，同时用图卷积网络介入尝试完成推理过程，并在该任务上达到了 state of the art，这些都是不错的亮点。但是稍有不足之处在于采用图卷积网络所做的『推理』仍然是功能非常有限，更多只是在基于 LSTM 的特征提取上做了一个分类工作，并没有太看到利用图的联通特性进行更多有效的推理过程，以及在第一部分处理过程中也变相的进行了知识库中的事实筛选，且其中部分筛选器目前还是独立于整个系统之外的，无法实现真正意义上的端到端学习训练。
     
     - **Visual_Genome**
     
       - 解析：[Visual Genome数据集梳理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/102403048)
     
       - [Visual Genome（VG）](https://link.zhihu.com/?target=http%3A//visualgenome.org/)是斯坦福大学李飞飞组于2016年发布的大规模图片语义理解数据集，他们希望该数据集能像ImageNet那样推动图片高级语义理解方面的研究。
     
         在视觉关系检测（VRD）的研究中，VG几乎成了的标准数据集，然而，该数据集的许多缺陷也一直饱受诟病，当前的VRD研究者实在很有必要将该数据集的构建过程梳理清楚。
     
         简单地概括，VG数据集主要由4个部分组成：
     
         - **Region Description**：图片被划分成一个个region，每个region都有与其对应的一句自然语言描述。
         - **Region Graph**：每个region中的object、attribute、relationship被提取出来，构成局部的“Scene Graph”。
         - **Scene Graph**：把一张图片中的所有Region Graph合并成一个全局的Scene Graph。
         - **QA**：每张图片会有多对QA，分为两种类型：region-based和freeform。前者基于Region Description提出，与局部region的内容直接相关；后者则基于整张图片来提出。
     
     - **zs-f-vqa**
     
       - 将外部知识纳入视觉问答（VQA）已成为一项重要的实际需求。现有方法大多采用流水线方法，其中包括不同组件用于知识匹配和提取、特征学习等。然而，这种流水线方法在某些组件表现不佳时容易受到影响，导致错误级联和整体性能不佳。此外，现有方法大多忽视了答案偏差问题，即在实际应用中，许多答案可能在训练过程中从未出现（即未见答案）。为了弥合这些差距，在本文中，我们提出了一种使用知识图谱和基于掩码的学习机制的零样本VQA算法，以更好地纳入外部知识，并针对F-VQA数据集提出了新的基于答案的零样本VQA划分。实验证明，我们的方法在零样本VQA中可以实现最先进的性能，同时在普通F-VQA任务上显著增强了现有的端到端模型。
     
       这一段可以看看[【论文小综】基于外部知识的VQA（视觉问答） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/389798289)

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

   - **Agent Sims: An Open-Source Sandbox for Large Language Model Evaluation**

        - Abstract: 随着类似于ChatGPT的大型语言模型（LLM）在社区中盛行，如何评估LLM的能力是一个悬而未决的问题。现有的评估方法存在以下缺点：（1）评估能力受限，（2）基准测试易受攻击，（3）评估指标不客观。我们建议采用基于任务的评估方法，即LLM代理在模拟环境中完成任务，这是解决上述问题的一劳永逸的解决方案。我们提出了AgentSims，这是一个易于使用的基础设施，供所有学科的研究人员测试他们感兴趣的特定能力。研究人员可以通过在交互式GUI上添加代理和建筑来构建评估任务，也可以通过几行代码部署和测试新的支持机制，例如记忆、规划和工具使用系统。我们的演示可在 [https://agentsims.com](https://agentsims.com/) 上访问。

        - **AgentSims**是一个开放式、自定义的沙盒评估基建设施。并且它还贴心地设置了专业开发者模式和普通用户模式，对那些不那么精通大模型和代码的研究人员还挺友好。具体操作中，研究人员可以**通过GUI**（交互式图形用户界面）**添加AI角色、建筑物**来设置不同的评估任务。此外，也可以通过少量代码来部署和测试新的支持机制，比如规划和记忆系统。并且可以设置角色的长期记忆，以及规划能力。

        - 众所周知，大模型对对话的上下文窗口有限制性。

          在AgentSims这里，团队用了一种辅助系统，让AI角色直追人类水平。

          具体抽象为三个部分：

          - **计划系统**，用来将目标分解为子任务，并记录当前状态，生成下一步行动。
          - **记忆系统**，将每日经历编码为向量，存储在数据库中，用于维持行为一致性。
          - **工具使用系统**，存储了使用设备的反馈，可以推断设备的操作方法。

          因为这个机制的存在，AgentSims的灵活度更高，用户可以自己“排列组合”不同的计划、记忆和学习系统。

   - **Emergent Social Learning via Multi-agent Reinforcement Learning**

        - 这篇文章的重点是 “Social Learning”，促使智能体以与其他智能体交互的形式学习。
        - Abstract: 社会学习是人类和动物智能的关键组成部分。通过从其环境中专家的行为中获取线索，社会学习者可以获得复杂的行为并迅速适应新的环境。本文调查了在多智能体环境中，独立的强化学习（RL）代理是否能够学会利用社会学习来提高其性能。我们发现，在大多数情况下，普通的无模型RL代理不会使用社会学习。我们分析了这种不足的原因，并表明通过对训练环境施加约束并引入基于模型的辅助损失，我们能够获得通用的社会学习策略，使代理能够：i）发现复杂的技能，这些技能不是通过单一代理的训练获得的，并且ii）通过从新环境中的专家获取线索，在线适应新的环境。相比之下，通过无模型RL或模仿学习训练的代理的泛化能力较差，并且在转移任务中不成功。通过混合多智能体和单独训练，我们可以获得使用社会学习获取技能的代理，这些技能代理在独处时也能够发挥作用，甚至胜过从一开始就单独训练的代理。
        - 引入了 “Social Learning with Auxiliary Losses”: We refer to our approach as social learning with an auxiliary predictive loss, and hypothesize that it will improve agents’ ability to learn from the cues of experts.
        - Agents can achieve high rewards by efficiently traversing between the goals in the correct order, and skilled agents are able to identify the correct order while incurring minimal
          penalties. Agents can accomplish this either by trial and error, or by observing other agents. In practice, since the behavior of other agents can be unpredictable and potentially non-stationary, agents more easily learn to solve the task directly through trial and error. But by adjusting the penalty for navigating to goals in the wrong order, we can penalize individual exploration and thereby encourage social learning.

   - **Humanoid Agents: Platform for Simulating Human-like Generative Agents**
        - 这个评估模拟系统加入了需求层次来让AI以人类的逻辑进行行动。
        - Abstract: 就像对原子、分子和细胞的计算机模拟塑造了我们对科学的研究方式一样，真实模拟人类样式的代理可以成为研究人类行为的有价值工具。我们提出了Humanoid Agents，这是一个指导生成式代理更像人类行为的系统，通过引入系统处理的三个元素：基本需求（例如饥饿、健康和能量）、情感和人际关系的亲密度。Humanoid Agents能够利用这些动态元素来调整他们的日常活动（Activity Planning）和与其他代理的对话（Dialogue Generation），这得到了实证实验的支持。我们的系统设计成可扩展到各种设置，我们演示了其中的三种，并且扩展到影响人类行为的其他元素（例如同理心、道德价值观和文化背景）。我们的平台还包括一个Unity WebGL游戏界面用于可视化，以及一个交互式分析仪表板，显示代理随时间的状态。我们的平台可在 https://www.humanoidagents.com/ 上获得，并且代码在 https://github.com/HumanoidAgents/HumanoidAgents 上。
   - **Social diversity and social preferences in mixed-motive reinforcement learning**
        - Abstract: 最近关于在纯冲突和纯共同利益游戏中的强化学习的研究强调了种群异质性的重要性。相比之下，在混合动机游戏中的强化学习研究主要采用了同质化的方法。鉴于混合动机游戏的定义特征——群体成员之间激励的不完全相关性，我们研究了种群异质性对混合动机强化学习的影响。我们借鉴社会心理学中的相互依赖理论，并赋予强化学习代理人社会价值取向（Social Value Orientation, SVO），这是对群体结果分布的偏好的灵活形式化。随后，我们探讨了SVO多样性对两个混合动机马尔可夫游戏中的强化学习代理人群体的影响。我们证明了SVO的异质性在代理人之间产生了有意义且复杂的行为变化，类似于相互依赖理论所建议的那样。在这些混合动机困境中的实证结果表明，在异质种群中训练的代理人相对于在同质种群中训练的代理人，发展出了特别泛化、高性能的策略。
        - 这篇没太看懂，因为还不太懂强化学习。这篇工作似乎是用多人游戏来验证的。
        - AGENTS: Multi-agent reinforcement learning and Markov games, Social Value Oriantation ...
        - 引入了 SVO 来 prefer certain group reward distributions between self and others.
        - MIXED-MOTIVE GAMES: Intertemporal social dilemmas，验证时使用了多人游戏，we consider two temporally and spatially extended mixed-motive games played with group size n = 5