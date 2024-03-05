---
layout: post
title:  "CV算法笔记"
date:   2024-03-05 00:28:00 +0800
categories: posts
---

### Selective Search

## 选择搜索算法（Selective Search）

[Reference](https://blog.csdn.net/qq_29695701/article/details/100669687)

> 原论文：
>
> 关键字：分层分组算法、初始化区域集、相似度计算
>
> *keywords: Hierarchical Grouping Algorithm, Obtaining Initial Regions, Calculating Similarity*

#### 分层分组算法（Hierarchical Grouping Algorithm）

selective search的主要内容。

**Input**: (color) image

**Output**: Set of object location hypotheses L

Obtain initial regions $R = \{r_1, \ldots, r_n\}$ using [Graph-Based Image Segmentation](https://blog.csdn.net/guoyunfei20/article/details/78727972)

Initialise similarity set $S = \Phi$

**foreach** *Neighbouring region pair* $(r_i, r_j)$ **do**

​	Calculate similarity $s(r_i, r_j)$ 

​	$S = S \cup s(r_i, r_j)$

**while** $S \neq \Phi$ **do**

​	Get highest similarity $s(r_i, r_j) = \mathbf{max}(S)$ 

​	Merge corresponding regions $r_t = r_i \cup r_j$

​	Remove similarities regarding $r_i : S \setminus s(r_i, r_*)$

​	Remove similarities regarding $r_i : S \setminus s(r_*, r_j)$

​	Calculate similarity set $S_t$ between $r_t$ and its neighbours

​	$S = S \cup S_t$

​	$R = R\cup r_t$

Extract object location boxes $L$ from all regions in $R$

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

### 基于图的图像分割（Graph-Based Image Segmentation）

[reference](https://blog.csdn.net/guoyunfei20/article/details/78727972)

#### 介绍

基于图的图像分割（Graph-Based Image Segmentation），论文《Efficient Graph-Based Image Segmentation》，P. Felzenszwalb, D. Huttenlocher，International Journal of Computer Vision, Vol. 59, No. 2, September 2004

论文下载和论文提供的C++代码在[这里](https://cs.brown.edu/people/pfelzens/segment/)。

Graph-Based Segmentation是经典的图像分割算法，其作者Felzenszwalb也是提出DPM（Deformable Parts Model）算法的大牛。

Graph-Based Segmentation算法是基于图的贪心聚类算法，实现简单，速度比较快，精度也还行。不过，目前直接用它做分割的应该比较少，很多算法用它作垫脚石，比如Object Propose的开山之作《Segmentation as Selective Search for Object Recognition》就用它来产生过分割（over segmentation）。

论文中，初始化时**每一个像素点都是一个顶点**，然后逐渐合并得到一个区域，确切地说是连接这个区域中的像素点的一个**MST（最小生成树）**。如下图，棕色圆圈为顶点，线段为边，合并棕色顶点所生成的MST，对应的就是一个分割区域。**分割后的结果其实就是森林**。

<p><img src="{{site.url}}/images/GBS1.png" width="20%" align="middle" /></p>

#### 相似性

既然是聚类算法，那应该依据何种规则判定何时该合二为一，何时该继续划清界限呢？对于孤立的两个像素点，所不同的是灰度值，自然就用灰度的距离来衡量两点的相似性，本文中是使用RGB的距离，即

当然也可以用perceptually uniform的Luv或者Lab色彩空间，对于灰度图像就只能使用亮度值了，此外，还可以先使用纹理特征滤波，再计算距离，比如先做Census Transform再计算Hamming distance距离。

#### **全局阈值 >> 自适应阈值，区域的类内差异、类间差异**

上面提到应该用亮度值之差来衡量两个像素点之间的差异性。对于两个区域（子图）或者一个区域和一个像素点的相似性，最简单的方法即只考虑连接二者的边的不相似度。如下图，已经形成了棕色和绿色两个区域，现在通过紫色边来判断这两个区域是否合并。那么我们就可以设定一个阈值，当两个像素之间的差异（即不相似度）小于该值时，合二为一。迭代合并，最终就会合并成一个个区域，效果类似于区域生长：星星之火，可以燎原。

<p><img src="{{site.url}}/images/GBS2.png" width="20%" align="middle" /></p>

**举例说明**：

<p><img src="{{site.url}}/images/GBS3.png" width="40%" align="middle" /></p>

对于上右图，显然应该聚成上左图所示的3类：高频区h，斜坡区s，平坦区p。

如果我们设置一个全局阈值，那么如果h区要合并成一块的话，那么该阈值要选很大，但是那样就会把p和s区域也包含进来，分割结果太粗。如果以p为参考，那么阈值应该选特别小的值，那样的话p区是会合并成一块，但是h区就会合并成特别特别多的小块，如同一面支离破碎的镜子，分割结果太细。显然，全局阈值并不合适，那么自然就得用自适应阈值。对于p区该阈值要特别小，s区稍大，h区巨大。

先来两个定义，原文依据这两个附加信息来得到自适应阈值。
一个区域内的类内差异 $\mathrm{Int}(C)$：

可以近似理解为一个区域内部最大的亮度差异值，定义是MST中不相似度最大的一条边。
俩个区域的类间差异 $\mathrm{Diff}(C1, C2)$：


即连接两个区域所有边中，不相似度最小的边的不相似度，也就是两个区域最相似的地方的不相似度。

直观的判断，当：

$$\mathrm{Int}(C) \geq \mathrm{Diff}(C1, C2)$$ 

时，两个区域应当合并。

#### 算法步骤

1. 计算每一个像素点与其8邻域或4邻域的不相似度；
2. 将边按照不相似度non-decreasing排列（从小到大）排序得到***e1, e2, ..., en***；
3. 选择***ei***；
4. 对当前选择的边**ej**（vi和vj不属于一个区域）进行合并判断。设其所连接的顶点为***(vi, vj)***，
5. ***if*** 不相似度小于二者内部不相似度
   - 更新阈值以及类标号
6. ***else***
   - 则按照排好的顺序，选择下一条边转到 ***Step 4***，否则结束。
