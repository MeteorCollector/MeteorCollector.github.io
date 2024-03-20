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

将单词的词 Embedding 和位置 Encoding 相加，就可以得到单词的表示向量 $x$，$x$ 就是 Transformer 的输入。

### Reference

[Transformer 修炼之道（一）、Input Embedding](https://zhuanlan.zhihu.com/p/372279569)

[Transformer模型详解](https://zhuanlan.zhihu.com/p/338817680)