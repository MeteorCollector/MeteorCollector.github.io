---
layout: post
title:  "CV论文阅读笔记"
date:   2024-03-05 00:28:00 +0800
categories: posts
---

### Selective Search

[Reference](https://blog.csdn.net/qq_29695701/article/details/100669687)

> 原论文：[Selective Search for Object Recognition](https://disi.unitn.it/~uijlings/selectiveSearchDraft.pdf)
> 关键字：分层分组算法、初始化区域集、相似度计算
> *keywords: Hierarchical Grouping Algorithm, Obtaining Initial Regions, Calculating Similarity*

#### 分层分组算法（Hierarchical Grouping Algorithm）

selective search的主要内容。

> **Input**: (color) image
>
> **Output**: Set of object location hypotheses L
>
> Obtain initial regions $R = \{r_1, \ldots, r_n\}$ using [Graph-Based Image Segmentation](https://blog.csdn.net/guoyunfei20/article/details/78727972)
>
> Initialise similarity set $S = \Phi$
>
> **foreach** *Neighbouring region pair* $(r_i, r_j)$ **do**
>
> ​	Calculate similarity $s(r_i, r_j)$ 
>
> ​	$S = S \cup s(r_i, r_j)$
>
> **while** $S \neq \Phi$ **do**
>
> ​	Get highest similarity $s(r_i, r_j) = \mathbf{max}(S)$ 
>
> ​	Merge corresponding regions $r_t = r_i \cup r_j$
>
> ​	Remove similarities regarding $r_i : S \setminus s(r_i, r_*)$
>
> ​	Remove similarities regarding $r_i : S \setminus s(r_*, r_j)$
>
> ​	Calculate similarity set $S_t$ between $r_t$ and its neighbours
>
> ​	$S = S \cup S_t$
>
> ​	$R = R\cup r_t$
>
> Extract object location boxes $L$ from all regions in $R$
>

**算法具体解释**

1. 使用 [Efficient Graph-Based Image Segmentation](https://blog.csdn.net/guoyunfei20/article/details/78727972) 中的方法初始化区域集 $R$；
2. 计算 $R$ 中相邻区域的**相似度**，并以此构建相似度集 $S$；
3. 如果 $S$ 不为空，则执行以下7个子步骤，否则，跳至步骤4；
    - 获取 $S$ 中的最大值 $s( r_i, r_j )$；
    - 将 $r_i$ 与 $r_j$ 合并成一个新的区域 $r_t$；
    - 将 $S$ 中与 $r_i$ 有关的值 $s(r_i, r_*)$ 剔除掉；
    - 将 $S$ 中与 $r_j$ 有关的值 $s(r_*, r_j)$ 剔除掉；
    - 使用步骤2中的方法，构建 $S_t$，它是 $S$ 的元素与 $r_t$ 之间的相似度构成的集合；
    - 将 $S_t$ 中的元素全部添加到 $S$ 中；
    - 将 $r_t$ 放入 $R$ 中。
4. 将 $R$ 中的区域作为目标的位置框 $L$，这就是算法的执行结果。

#### 相似度计算

原论文里考虑了四个方面的相似度：颜色 $s_{colour}$ 、纹理 $s_{texture}$ 、尺度  $s_{size}$ 、空间交叠 $s_{fill}$ ，并将这四个相似度以线性组合的方式综合在一起，作为最终被使用的相似度 $s(r_i,r_j)$:

$$
s(r_i,r_j)=\alpha_1 s_{colour}(r_i,r_j) + \alpha_2 s_{texture}(r_i,r_j) + \alpha_3 s_{size}(r_i,r_j) + \alpha_4 s_{fill}(r_i,r_j)
$$


下面是这四种相似度的介绍。

**颜色相似度**

将区域的颜色空间转换为直方图，三个颜色通道的bins取25。于是我们可以得到某个区域 $r_i$ 的颜色直方图向量：$C_i=\{c^1_i,...,c^n_i\}$，其中 $n=75$（计算方式：$bins\times n\_channels=25\times3bins×n_channels=25×3$），并且$C_i$是用区域的 $L_1$ 范数归一化后的向量。关于 $r_t = r_i \cup r_j$ 的 $C_t$，计算方式是这样的：
$$
C_t=\frac{size(r_i)\times C_i + size(r_j)\times C_j}{size(r_i)+size(r_j)}
$$

而 $r_t$ 尺寸的计算方式为：$size(r_t)=size(r_i)+size(r_j)$。
颜色相似度的计算公式：
$$
s_{colour}(r_i,r_j)=\sum^{n}_{k=1}\textbf{min}(c^k_i,c^k_j)​
$$
**纹理相似度**

对每一个颜色通道，在8个方向上提取高斯导数 $\sigma=1$。在每个颜色通道的每个方向上，提取一个bins为10的直方图，从而得到每个区域 $r_i$ 的纹理直方图向量 $T_i=\{t^1_i,...,t^n_i\}$，其中 $n=240$（计算方式：$n\_orientations\times bins\times n\_channels=8\times10\times3$），$T_i$ 也是用区域的 $L_1$ 范数归一化后的向量。
纹理相似度的计算公式：
$$
s_{texture}(r_i,r_j)=\sum^{n}_{k=1}\mathbf{min}(t^k_i,t^k_j)
$$
**尺度相似度**

尺度相似度的计算公式：
用于优先合并小区域。
$$
s_{size}(r_i,r_j)=1-\frac{size(r_i)+size(r_j)}{size(im)}
$$


其中，$size(im)$是整张图片的像素级的尺寸。

**空间交叠相似度**

用于优先合并被包含进其他区域的区域。
空间交叠相似度的计算公式：
$$
s_{fill}(r_i,r_j)=1-\frac{size(BB_{ij})-size(r_i)-size(r_j)}{size(im)}
$$
其中，$BB_{ij}$  是能够框住 $r_i$ 与 $r_j$ 的最小矩形框。

### Graph-Based Image Segmentation