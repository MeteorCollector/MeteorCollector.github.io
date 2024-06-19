---
layout: post
title:  "机器学习导论复习"
date:   2024-06-18 14:28:00 +0800
categories: posts
tag: ml
---

## 写在前面

忙活了这么长时间本校夏令营，砸了这么多精力，又熬大夜又肠胃炎，最后还是本校wl了，而且要被jyy骂菜。人生中从未有过如此屈辱的时刻。

不过期末考试还是要复习。我把机器学习导论的复习内容整理到这里，大概就是过一遍作业吧。

有时间一定要把个人站的风格改一下，现在这个风格对于笔记来说太不友好了。

## 数学基础

### 矩阵论

#### 梯度 gradient:

`$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(\mathbf{x}) \\ \vdots  \\ \frac{\partial f}{\partial x_d}(\mathbf{x}) \end{bmatrix}$$`

#### Hessian Matrix:

`$$\nabla^2 f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1 \partial x_1}(\mathbf{x}) & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_d}(\mathbf{x})\\
\vdots  & \ddots  & \vdots \\
\frac{\partial^2 f}{\partial x_d \partial x_1}(\mathbf{x})  & \cdots & \frac{\partial^2 f}{\partial x_d \partial x_d}(\mathbf{x})
\end{bmatrix}$$`

#### 矩阵求导

[矩阵求导公式的数学推导（矩阵求导——基础篇） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/273729929)

记 $\mathbf{x}$ 是列向量，有

`$$\frac{\partial(\mathbf{x^\mathrm{T}a})}{\partial \mathbf{x}} = \frac{\partial{\mathbf{a^\mathrm{T}x}}}{\partial\mathbf{x}} = \mathbf{a}$$`

### 概率论

#### 卡方分布

若有 `$\mathrm{i.i.d.}\quad x_1, x_2, \cdots x_n \sim \mathcal{N}(0, 1)$`

`$$\chi^2(n) = \sum^n_i x_i^2$$`

#### 学生t分布

假设 `$X \sim \mathcal{N}(0, 1)$` 且 `$Y \sim \chi^2(n)$` 且两者独立，那么 `$Z = \frac{X}{\sqrt{Y / n}}$` 的分布就是自由度为 $n$ 的学生T分布。


### 优化

- 对于只有等式约束的优化问题，可以直接用拉格朗日乘子法列出拉格朗日函数，将其转化为无约束优化问题求解
- 对于包含不等式约束的优化问题，仍然可以像只有等式约束时一样列出拉格朗日函数，但此时函数中会包含对拉格朗日乘子的新约束，优化它得到的最优值结果一定满足 KKT 条件（KKT 是取最优参数值的必要条件，对于某些特殊的凸优化问题是充要条件）
- 含有不等式约束的问题列出拉格朗日函数后仍有约束不好处理，这时我们可以将其转化为拉格朗日对偶问题，这个对偶问题一定是凸优化问题，因此易于求解。优化问题一定具有弱对偶性，但要想对偶问题和原问题同解其必须满足强对偶性，强对偶性的充分条件是Slater 条件，必要条件是 KKT 条件

#### KKT 条件

[约束优化方法之拉格朗日乘子法与KKT条件 - ooon - 博客园 (cnblogs.com)](https://www.cnblogs.com/ooon/p/5721119.html)

对于形式化的不等式约束优化问题

`$$\underset{x}{\min} f(x) \\ s.t.\; h_i(x) = 0,\; i = 1, 2, \ldots, m \\ \quad \quad  g_i(x) \leq 0, \; j = 1, 2, \ldots, n$$`

列出拉格朗日方程

`$$L(x, \alpha, \beta) = f(x) + \sum^m_{i=1}\alpha_ih_i(x) + \sum^n_{j=1}\beta_ig_i(x)$$`

可行解 $x$ 需要满足以下 KKT 条件：

`$$\begin{aligned}\nabla_xL(x, \alpha, \beta) & = 0 &(1) \\ \beta_j g_j (x) & = 0, \;j = 1, 2, \ldots, n & (2) \\ h_i(x) &= 0,\; i = 1, 2, \ldots ,m & (3) \\ g_j(x) & \leq 0, \; j = 1, 2, \ldots ,n &(4) \\ \beta_j &\geq 0, \; j = 1, 2, \ldots ,n & (5)\end{aligned}$$`

**满足 KKT 条件后极小化 Lagrangian 即可得到在不等式约束条件下的可行解。** KKT 条件看起来很多，其实很好理解:

(1) ：拉格朗日取得可行解的必要条件；

(2) ：这就是以上分析的一个比较有意思的约束，称作松弛互补条件；

(3) ∼ (4) ：初始的约束条件；

(5) ：不等式约束的 Lagrange Multiplier 需满足的条件。

主要的KKT条件便是 (3) 和 (5) ，只要满足这两个条件便可直接用拉格朗日乘子法。

#### 费马最优条件 Fermat's Optimality Condition

其实就是

`$$\mathbf{x}^* \in \arg \min \{f(\mathbf{x} \mid x \in \mathbb{R}^d)\}$$`

iff `$0 \in \partial f(\mathbf{x}^*)$`

## 模型评估方法

评估方法

- 留出法(hold-out)：划出不相交的训练集和测试集
- 交叉验证法(cross validation)：划分为k个子集，用k-1个训练，用1个验证(k-fold)
- 自助法(bootstrap)：有放回的采样

## 模型评价标准

精度(Accuracy) = 预测正确的样本数 / 总样本数 = 1 - ErrorRate

错误率(Error Rate) = 预测错误的样本数 / 总样本数 = 1 - Accuracy

查准率(Precision) = 真阳性 / (真阳性 + 假阳性) = 对于二分类模型，正确的阳性占所有查出来阳性的比例

查全率(Recall) = 真阳性 / (真阳性 + 假阴性) = 对于二分类模型，正确的阳性占所有真正阳性的比例

宏查准率(Macro Presision) = (1 / n) * (类别1的查准率 + 类别2的查准率 + ... + 类别n的查准率)

微查准率(Micro Precision) = (类别1的真阳性 + 类别2的真阳性 + ...... + 类别n的真阳性) / (类别1查出来的阳性 +  类别2查出来的阳性 + ...... + 类别n查出来的阳性)

宏查全率(Macro Recall) = (1 / n) * (类别1的查全率 + 类别2的查全率 + ... + 类别n的查全率)

微查全率(Micro Recall) = (类别1的真阳性 + 类别2的真阳性 + ...... + 类别n的真阳性) / (类别1真正的阳性 +  类别2真正的阳性 + ...... + 类别n真正的阳性)

宏F1度量(Macro F1) = (2 * Macro-P * Macro-R) / (Macro-P + Macro-R)

微F1度量(Micro F1) = (2 * Micro-P * Micro-R) / (Micro-P + Micro-R)

**ROC**：受试者工作特征(Receiver Operating Characteristic)

根据学习器的预测结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，纵轴为真正例率 TPR = TP / (TP + FN)，横轴为正例率 FPR = FP / (TN + FP)。

**AUC**：Area Under Curve

若ROC曲线上坐标分别为 `$\{(x_1, y_1), (x_2, y_2), \cdots , (x_m, y_m)\}$`

`$$\mathbf{AUC} = \frac{1}{2} \sum^{m-1}_{i=1} (x_{i+1} - x_i) \cdot (y_i + y_{i+1})$$`

`$$l_{rank} = \frac{m^+}{m^-} \sum_{x^+ \in D^+} \sum_{x^- \in D^-} \left(\mathbb{I} (f(x^+) < f(x^-)) + \frac{1}{2}\mathbb{I} (f(x^+) = f(x^-))\right)$$`

`$$\mathbf{AUC} = 1 - l_{rank}$$`

`$$\mathbf{AUC} = \frac{m^+}{m^-} \sum_{x^+ \in D^+} \sum_{x^- \in D^-} \left(\mathbb{I} (f(x^+) > f(x^-)) + \frac{1}{2}\mathbb{I} (f(x^+) = f(x^-))\right)$$`

## 线性回归

对于多元线性回归

`$$f(\mathbf{x}_i) = \mathbf{w}^\mathrm{T}\mathbf{x}_i + b \simeq y_i$$`

为了便于讨论，将常数项吸收，记

`$$\mathbf{X} = \begin{pmatrix}
 \mathbf{x}_1^\mathrm{T} & 1\\
 \mathbf{x}_2^\mathrm{T} & 1\\
 \vdots & \vdots \\
 \mathbf{x}_1^\mathrm{T} & 1
\end{pmatrix}$$`

我们想要的是

`$$\hat{\mathbf{w}}^* = \underset{\mathbf{\hat{w}}}{\arg \min} (\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})^\mathrm{T}(\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})$$`

对 `$\mathbf{\hat{w}}$` 求偏导并令式子为0，得

`$$\mathbf{\hat{w}}^* = (\mathbf{X}^\mathrm{T}\mathbf{X})^{-1} \mathbf{X}^\mathrm{T} \mathbf{y}$$`

额外地，对数线性回归为

`$$\ln y = \mathbf{w}^\mathrm{T} \mathbf{x} + b$$`

## LDA 线性判别分析

同类投影尽可能近，异类投影尽可能远

TODO: PPT上内容

## PCA 主成分分析

最大化类别无关的全局散度

详细过程见 hw 3-1

注意，这里样本矩阵 $\mathbf{X}$ 的行数对应数据的维度，列数对应数据的组数。也就是说每个数据对应一个**列向量**。

- 计算均值，标准差
- 获得标准化后的样本矩阵 $\mathbf{X}_{\mathrm{std}}$ （减去均值，除以标准差）
- 获得协方差矩阵 `$\mathbf{S} = \frac{1}{n - 1} \mathbf{X}_{\mathrm{std}} \mathbf{X}_{\mathrm{std}}^\mathrm{T}$`
- 计算协方差矩阵的特征值，即计算 $\left|\mathbf{S} - \lambda\mathbf{I}\right| = 0$ 的解，记为 $\lambda_1, \lambda_2, \ldots , \lambda_n$
- 假若降至 $m$ 维，取最大的 $m$ 个特征值所对应的单位特征向量 $w_1, w_2, \ldots, w_m$ （注意，求出特征向量之后还需要归一化。特征向量即 $\left(\mathbf{S} - \lambda\mathbf{I}\right)\mathbf{x} = 0$ 的解，再等比例缩放为**单位向量**）
- 投影矩阵 $\mathbf{W} = (w_1, w_2, \ldots , w_m)$
- $\mathbf{X}_\mathrm{new} = \mathbf{W}^\mathrm{T}\mathbf{X}_{\mathrm{std}}$

在选择压缩到的维度时，可以设计一个重构阈值 $t$，然后选择使下式成立的最小 $d^\prime$ 值：

`$$\frac{\sum^{d^\prime}_{i=1}\lambda_i}{\sum^{d}_{i=1}\lambda_i} \geq t$$`

TODO: PPT上内容

## 决策树

#### 信息增益

假定当前样本集合 $D$ 中第 $k$ 类样本所占的比例为 $p_k (k = 1, 2, \ldots , \left| \mathcal{Y} \right|)$，则 $D$ 的信息熵定义为

`$$\mathrm{Ent}(D) = - \sum^{\left| \mathcal{Y} \right|}_{k=1} p_k \log_2 p_k$$`

假设某次划分后从**节点 $D$** 产生了 $V$ 个分支节点，第 $v$ 个分支结点包含了 $D$ 中所有在属性 $a$  上取值为 $a^v$ 的样本，记为 $D^v$，则信息增益 (information gain)定义为

`$$\mathrm{Gain}(D, a) = \mathrm{Ent}(D) - \sum^{V}_{v=1}\frac{\left|D^v\right|}{\left|D\right|}\mathrm{Ent}(D^v)$$`

也就是对分类后新分类的信息熵进行加权。注意每次都是只考虑要分开的节点，并不是对全局做运算。

#### 增益率

信息增益对可取值数目较多的属性有偏好。为了减少这种不利影响，增益率定义为

`$$\mathrm{Gain\_ratio}(D, a) = \frac{\mathrm{Gain(D, a)}}{\mathrm{IV}(a)}$$`

其中

`$$\mathrm{IV} = - \sum^{V}_{v=1} \frac{\left|D^v\right|}{\left|D\right|}\log_2 \frac{\left|D^v\right|}{\left|D\right|}$$`

 #### 基尼系数

数据集的纯度可以用基尼值度量：

`$$\mathrm{Gini}(D) = \sum^{\left|\mathcal{Y}\right|}_{k=1}\sum_{k^\prime \neq k}p_k p_{k^\prime} = 1 - \sum^{\left| \mathcal{Y} \right|}_{k=1}p^2_k$$`

$\mathrm{Gini}(D)$ 越小，数据集 $D$ 的纯度越高。

属性 $a$ 的基尼指数定义为

`$$\mathrm{Gini\_index}(D, a) = \sum^V_{v=1}\frac{\left|D^v\right|}{\left|D\right|}\mathrm{Gini}(D^v)$$`

用基尼指数构建决策树时，选择划分后基尼指数最小的属性作为最优划分属性。

#### 剪枝

“预剪枝”：在决策树生成过程中，对每个结点在划分前先进行估计，若当前节点的划分不能带来决策树泛化性能提升，则停止划分并将当前节点标记为叶节点；

“后剪枝”：决策树生成完成后自底向上对非叶节点进行考察，若将该结点对应的子树替换成叶节点能带来泛化性能提升，则进行替换。（训练时间开销大）

在西瓜书上，两者的考察标准都是在验证集上的精度(precision)。

## 神经网络 Neural Networks

#### 基础原理

首先, 考虑一个多层前馈神经网络, 规定网络的输入层是第 $0$ 层, 输入为 $\mathbf{x} \in \mathbb{R}^d$. 网络有 $M$ 个隐层, 第 $h$ 个隐层的神经元个数为 $N_h$, 输入为 $\mathbf{z}_h\in \mathbb{R}^{N_{h-1}}$, 输出为 $\mathbf{a}_h \in \mathbb{R}^{N_h}$, 权重矩阵为 $\mathbf{W}_h \in \mathbb{R}^{N_{h-1} \times N_{h}}$, 偏置参数为 $\mathbf{b}_h \in \mathbb{R}^{N_h}$. 网络的输出层是第 $M+1$ 层, 神经元个数为 $C$, 权重矩阵为 $\mathbf{W}_{M+1} \in \mathbb{R}^{N_M \times C}$, 偏置参数为 $\mathbf{b}_{M+1} \in \mathbb{R}^C$, 输出为 $\mathbf{y} \in \mathbb{R}^C$. 网络隐层和输出层的激活函数均为 $f$, 网络训练时的损失函数为 $\mathcal{L}$, 且 $f$ 与 $\mathcal{L}$ 均可微.

有

`$$\begin{aligned}
\mathbf{z}_h & = \mathbf{W}_h^\mathrm{T} \mathbf{a}_{h-1}, & (1 \leq h \leq M) \\
\mathbf{a}_h & = f(\mathbf{z}_h + \mathbf{b}_h), & (1 \leq h \leq M) \\
\mathbf{y} & = f(\mathbf{W}^\mathrm{T}_{M+1} \mathbf{a}_M + \mathbf{b}_{M+1})
\end{aligned}$$`

#### 防止过拟合

- 正则化 **TODO**
- 数据增强
- Dropout
- Early Stopping

## 支持向量机 Support Vector Machine

#### 对偶问题

#### 核函数

另 $\mathcal{X}$ 为输入空间，$\kappa(\cdot,\cdot)$ 时定义在 $\mathcal{X} \times \mathcal{X}$ 上的对称函数，则 $\kappa$ 是核函数当且仅当对于任意数据 $D = \{\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_m\}$，“核矩阵” $\mathbf{K}$ 总是半正定的：

`$$\mathbf{K} = \begin{bmatrix}
\kappa(\mathbf{x}_1, \mathbf{x}_1) & \cdots & \kappa(\mathbf{x}_1, \mathbf{x}_j) & \cdots & \kappa(\mathbf{x}_1, \mathbf{x}_m)\\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\kappa(\mathbf{x}_i, \mathbf{x}_1) &  & \kappa(\mathbf{x}_i, \mathbf{x}_j) &  & \kappa(\mathbf{x}_i, \mathbf{x}_m)\\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\kappa(\mathbf{x}_m, \mathbf{x}_1) & \cdots & \kappa(\mathbf{x}_m, \mathbf{x}_j) & \cdots & \kappa(\mathbf{x}_m, \mathbf{x}_m)
\end{bmatrix}$$`