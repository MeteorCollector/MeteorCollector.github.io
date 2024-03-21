---
layout: post
title:  "CV概念（五）: Transformer专辑"
date:   2024-03-20 11:03:00 +0800
categories: posts
tag: cv
---

## 写在前面

注意力机制在NLP领域和CV领域都有重要的应用，所以专门整理一下这边的知识点。

[Paper: Attention is all You Need](https://arxiv.org/abs/1706.03762)

另外，也需要贴一下transformet在cv这边重要应用论文：

[DETR: End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)

## 基本结构

<p><img src="{{site.url}}/images/transformer.png" width="75%" align="middle" /></p>

### Word Embedding

> we use learned embeddings to convert the input
tokens and output tokens to vectors of dimension $d_{model}$. 

在cv的目标识别中往往使用one-hot来标注不同类别的目标。但是one-hot会造成极大的空间浪费——要知道在NLP中，词汇量是非常大的。

因此，我们需要另一种词的表示方法，能够体现词与词之间的关系，使得意思相近的词有相近的表示结果，这种方法即 Word Embedding——核心是设计一个**可学习的权重矩阵** $\mathbf{W}$，将**词向量**与这个矩阵进行点乘，即得到新的表示结果。

### Position Encoding

接下来要加入位置信息。有公式（PE即Position Encoding）：

$$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$$

将单词的词 Embedding 和位置 Encoding 相加，就可以得到单词的表示向量 $X$，$X$ 就是 Transformer 的输入。

### Self-Attention

这一部分是 Transformer 比较核心的内容。十分推荐李宏毅老师的讲解：[Youtube](https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=60)

经过上一个步骤，我们获得了矩阵 $X$。其中，$X$的每一个行向量都表示一个单词进行 embed 之后的结果。

```
input X

embedded word 1
embedded word 2
embedded word 3
```

对于每一个单词，都生成一个 $q$(query) 向量、$k$(key) 向量和一个 $v$(value) 向量。$q$向量组成的矩阵$Q$和$k$向量组成的矩阵$K$进行运算，代表每个word对每个word的key进行query。这一步运算的公式是：

$$\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

由于矩阵乘法的特性，每个$q$都能和每个$k$进行运算，因此最后的计算结果带有全局的信息，也就是获取了全部的上下文。这一点是十分重要的。

关于这个神秘的$\sqrt{d_k}$，作者是这样说的：

> We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

也就是说防止 $Q$ 和 $K$ 的点积值过大，softmax之后梯度太小。只是一个约定而已。

然后再与 $v$ 组成的 $V$ 矩阵做运算，获得最终的算式：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

那么$Q$、$K$、$V$怎么获得呢？嗯那自然是学习出来的了。我们学习出三个线性变换矩阵 $W_Q$，$W_K$，$W_V$，然后用输入 $X$ 和它相乘：

$$X \times W_Q = Q$$

$$X \times W_K = K$$

$$X \times W_V = V$$

由此计算得到 $Q$，$K$，$V$ 再做上面的计算就可以了。j记这一步的结果为 $Z$，这个计算过程的操作就是 **Scaled Dot-Product Attention**。

### Multi-Head Attention

接下来就把多个 Self-Attention 拼接成一个 Multi-Head Attention。其实简单，直接把它们 concat 到一起再进行线性变换就可以了。下图展示了一种 $h = 8$ （8个Self-Attention）的情况。

<p><img src="{{site.url}}/images/multihead.png" width="75%" align="middle" /></p>

注意，Multi-Head Attention 的输出矩阵 $Z$ 和输入矩阵 $X$ 的维度是一样的。

### Add & Norm

<p><img src="{{site.url}}/images/addnorm.png" width="40%" align="middle" /></p>

接下来是这一部分。

Add & Norm 层的公式是这样的：

$$\mathrm{LayerNorm}(X + \mathrm{MultiHeadAttention(X)})$$

$$\mathrm{LayerNorm}(X + \mathrm{FeedForward(X)})$$

这一段是受到了残差神经网络的影响。

### Feed Forward

这一模块是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。

$$\max(0, XW_1 + b_1)W_2 + b_2$$

### Encoder

综上所述，一个 Encoder Block 中的所有组件都被介绍完毕了。

通过上面描述的 Multi-Head Attention, Feed Forward, Add & Norm 就可以构造出一个 Encoder block，Encoder block 接收输入矩阵 $X_{(n \times d)}$，并输出一个矩阵$O_{(n \times d)}$。通过多个 Encoder block 叠加就可以组成 Encoder。

第一个 Encoder block 的输入为句子单词的表示向量矩阵，后续 Encoder block 的输入是前一个 Encoder block 的输出，最后一个 Encoder block **输出的矩阵就是编码信息矩阵 C**，这一矩阵后续会用到 Decoder 中。

### Decoder

<p><img src="{{site.url}}/images/decoder.png" width="75%" align="middle" /></p>

接下来看向结构的另一边。

红色部分为 Transformer 的 Decoder block 结构，与 Encoder block 相似，但是存在一些区别：

- 包含两个 Multi-Head Attention 层。

- 第一个 Multi-Head Attention 层采用了 Masked 操作。

- 第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。

- 最后有一个 Softmax 层计算输出（比如下一个词的预测等等）。

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作。这是因为Sequence的分析是有序的。例如在翻译任务中，单词往往是顺序翻译的，即翻译完第 $i$ 个单词，才可以翻译第 $i+1$ 个单词。

在这之后在经过一层 Multi-Head Attention 利用全局信息，避免局限性；最后用 Softmax 进行输出。

这就是Transformer的结构了。

### Reference

[Transformer 修炼之道（一）、Input Embedding](https://zhuanlan.zhihu.com/p/372279569)

[Transformer模型详解](https://zhuanlan.zhihu.com/p/338817680)