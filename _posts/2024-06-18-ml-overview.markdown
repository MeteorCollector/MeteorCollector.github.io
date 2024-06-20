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

带 * 为作业中未出现但是 PPT 里出现。

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

#### 高斯（正态）分布

`$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp \left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$`

作业 5-3 用到了截断正态分布。

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

**NFL(No Free Lunch)定理** *： 一个算法 `$\mathfrak{L}_a$` 若在某些问题上比另一个算法 `$\mathfrak{L}_b$` 好，必存在另一些问题，`$\mathfrak{L}_b$` 比 `$\mathfrak{L}_a$ ` 好。

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

## 线性模型

### 线性回归

对于多元线性回归

`$$f(\mathbf{x}_i) = \mathbf{w}^\mathrm{T}\mathbf{x}_i + b \simeq y_i$$`

为了便于讨论，将常数项吸收，记

`$$\mathbf{X} = \begin{pmatrix}
 \mathbf{x}_1^\mathrm{T} & 1\\
 \mathbf{x}_2^\mathrm{T} & 1\\
 \vdots & \vdots \\
 \mathbf{x}_n^\mathrm{T} & 1
\end{pmatrix}$$`

我们想要的是

`$$\hat{\mathbf{w}}^* = \underset{\mathbf{\hat{w}}}{\arg \min} (\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})^\mathrm{T}(\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})$$`

对 `$\mathbf{\hat{w}}$` 求偏导并令式子为0，得

`$$\mathbf{\hat{w}}^* = (\mathbf{X}^\mathrm{T}\mathbf{X})^{-1} \mathbf{X}^\mathrm{T} \mathbf{y}$$`

额外地，对数线性回归为

`$$\ln y = \mathbf{w}^\mathrm{T} \mathbf{x} + b$$`

### 对数几率回归*

可以用线性回归解决二分类任务，此时训练集 $y \in \{0, 1\}$

线性回归模型产生的实值输出是 $z = \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b$，需要将其映射到 $[0, 1]$，往往使用logistic function：

`$$y = \frac{1}{1 + \mathrm{e}^{-z}}$$`

即

`$$\ln \frac{1}{1-y} = \boldsymbol{w}^\mathrm{T} \boldsymbol{x} + b$$`

这被称作“对数几率”。

### LDA 线性判别分析

同类投影尽可能近，异类投影尽可能远

记第 $i$ 类示例的集合为 `$X_i$` ，均值向量为 `$\boldsymbol{\mu}_i$`，协方差矩阵为 `$\boldsymbol{\Sigma}_i$`

则两类样本的中心在支线上的投影是 `$\boldsymbol{w}^\mathrm{T} \boldsymbol{\mu}_0$` 和 `$\boldsymbol{w}^\mathrm{T} \boldsymbol{\mu}_1$`，两类样本在投影后的协方差分别为  `$\boldsymbol{w}^\mathrm{T} \boldsymbol{\Sigma}_0 \boldsymbol{w}$` 和 `$\boldsymbol{w}^\mathrm{T} \boldsymbol{\Sigma}_1 \boldsymbol{w}$`

我们的目标是使 `$\boldsymbol{w}^\mathrm{T} \boldsymbol{\Sigma}_0 \boldsymbol{w} + \boldsymbol{w}^\mathrm{T} \boldsymbol{\Sigma}_1 \boldsymbol{w}$` 尽可能小，`$\left\| \boldsymbol{w}^\mathrm{T} \boldsymbol{\mu}_0 - \boldsymbol{w}^\mathrm{T} \boldsymbol{\mu}_1 \right\|_2^2$` 尽可能大，即最大化

`$$J = \frac{\left\| \boldsymbol{w}^\mathrm{T} \boldsymbol{\mu}_0 - \boldsymbol{w}^\mathrm{T} \boldsymbol{\mu}_1 \right\|_2^2}{\boldsymbol{w}^\mathrm{T} \boldsymbol{\Sigma}_0 \boldsymbol{w} + \boldsymbol{w}^\mathrm{T} \boldsymbol{\Sigma}_1 \boldsymbol{w}} = \frac{\boldsymbol{w}^\mathrm{T}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^\mathrm{T}\boldsymbol{w}}{\boldsymbol{w}^\mathrm{T} (\boldsymbol{\Sigma}_0 + \boldsymbol{\Sigma}_1) \boldsymbol{w}}$$`

定义类内散度矩阵

`$$\boldsymbol{S}_w = \boldsymbol{\Sigma}_0 + \boldsymbol{\Sigma}_1 = \sum_{\boldsymbol{x} \in X_0} (\boldsymbol{x} - \boldsymbol{\mu}_0)(\boldsymbol{x} - \boldsymbol{\mu}_0)^\mathrm{T} + \sum_{\boldsymbol{x} \in X_1} (\boldsymbol{x} - \boldsymbol{\mu}_1)(\boldsymbol{x} - \boldsymbol{\mu}_1)^\mathrm{T}$$`

定义类间散度矩阵

`$$\boldsymbol{S}_b = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^\mathrm{T}$$`

目标重写为最大化广义瑞利商

`$$J = \frac{\boldsymbol{w}^\mathrm{T}\boldsymbol{S}_b\boldsymbol{w}}{\boldsymbol{w}^\mathrm{T}\boldsymbol{S}_w\boldsymbol{w}}$$`

之后是使用奇异值分解进行求解。

推广到多分类任务中......详见书/PPT

### PCA 主成分分析

最大化类别无关的全局散度

详细过程见 hw 3-1

注意，这里样本矩阵 $\mathbf{X}$ 的行数对应数据的维度，列数对应数据的组数。也就是说每个数据对应一个**列向量**。

- 计算均值，标准差
- 获得标准化后的样本矩阵 $\mathbf{X}_{\mathrm{std}}$ （减去均值，除以标准差）
- 获得协方差矩阵 `$\mathbf{S} = \frac{1}{n - 1} \mathbf{X}_{\mathrm{std}} \mathbf{X}_{\mathrm{std}}^\mathrm{T}$`
- 计算协方差矩阵的特征值，即计算 `$\left|\mathbf{S} - \lambda\mathbf{I}\right| = 0$` 的解，记为 `$\lambda_1, \lambda_2, \ldots , \lambda_n$`
- 假若降至 $m$ 维，取最大的 $m$ 个特征值所对应的单位特征向量 $w_1, w_2, \ldots, w_m$ （注意，求出特征向量之后还需要归一化。特征向量即 $\left(\mathbf{S} - \lambda\mathbf{I}\right)\mathbf{x} = 0$ 的解，再等比例缩放为**单位向量**）
- 投影矩阵 `$\mathbf{W} = (w_1, w_2, \ldots , w_m)$`
- `$\mathbf{X}_\mathrm{new} = \mathbf{W}^\mathrm{T}\mathbf{X}_{\mathrm{std}}$`

在选择压缩到的维度时，可以设计一个重构阈值 $t$，然后选择使下式成立的最小 $d^\prime$ 值：

`$$\frac{\sum^{d^\prime}_{i=1}\lambda_i}{\sum^{d}_{i=1}\lambda_i} \geq t$$`

TODO: PPT上内容

### 多分类任务*

若从二分上推广，有 OvO、OvR、MvM 几种方法，最常用的 MvM 技术是纠错输出码 (Error Correcting Output Codes, ECOC)

### 类别不平衡问题*

“再缩放”，$\frac{y^\prime}{1 - y^\prime} = \frac{y}{1-y} \times \frac{m^-}{m^+}$

## 决策树

#### 信息增益

假定当前样本集合 $D$ 中第 $k$ 类样本所占的比例为 `$p_k (k = 1, 2, \ldots , \left| \mathcal{Y} \right|)$`，则 $D$ 的信息熵定义为

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

#### 连续值问题*

基本思路：将连续属性离散化

#### 缺失值问题*

基本思路：样本赋权，权重划分

#### 轴平行划分/多变量决策树*

## 神经网络 Neural Networks

#### 基础原理

首先, 考虑一个多层前馈神经网络, 规定网络的输入层是第 $0$ 层, 输入为 `$\mathbf{x} \in \mathbb{R}^d$`. 网络有 $M$ 个隐层, 第 $h$ 个隐层的神经元个数为 `$N_h$`, 输入为 `$\mathbf{z}_h\in \mathbb{R}^{N_{h-1}}$`, 输出为 `$\mathbf{a}_h \in \mathbb{R}^{N_h}$`, 权重矩阵为 `$\mathbf{W}_h \in \mathbb{R}^{N_{h-1} \times N_{h}}$`, 偏置参数为 `$\mathbf{b}_h \in \mathbb{R}^{N_h}$`. 网络的输出层是第 $M+1$ 层, 神经元个数为 $C$, 权重矩阵为 `$\mathbf{W}_{M+1} \in \mathbb{R}^{N_M \times C}$`, 偏置参数为 `$\mathbf{b}_{M+1} \in \mathbb{R}^C$`, 输出为 `$\mathbf{y} \in \mathbb{R}^C$`. 网络隐层和输出层的激活函数均为 $f$, 网络训练时的损失函数为 $\mathcal{L}$, 且 $f$ 与 $\mathcal{L}$ 均可微.

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

#### BP算法*

#### RBF, SOM, 级联相关, Elman 网络, 深度学习......

## 支持向量机 Support Vector Machine

间隔 `$\gamma = \frac{2}{\left\| \boldsymbol{w} \right\|}$`

原形式

`$$\underset{\boldsymbol{w}, b}{\max} \frac{2}{\left\| \boldsymbol{w} \right\|} \\ \text{s.t.}\; y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i + b) \geq 1,\quad i = 1, 2, \ldots, m.$$`

等价为

`$$\underset{\boldsymbol{w}, b}{\min} \frac{1}{2}\left\| \boldsymbol{w} \right\|^2 \\ \text{s.t.}\; y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i + b) \geq 1,\quad i = 1, 2, \ldots, m.$$`

#### 对偶问题

对原式的每条约束添加拉格朗日乘子 `$\alpha_i \geq 0$`，则该问题的拉格朗日函数可写为

`$$L(\boldsymbol{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\left\|\boldsymbol{w}\right\|^2 + \sum^m_{i=1} \alpha_i (1 - y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i + b))$$ `

**注意正负**

其中 `$\boldsymbol{\alpha} = (\alpha_1;\alpha_2;\ldots;\alpha_m)$` ，令 `$L(\boldsymbol{w}, b, \boldsymbol{\alpha})$` 对 $\boldsymbol{w}$ 和 $b$ 偏导为零得

`$$\boldsymbol{w} = \sum^m_{i=1}\alpha_i y_i \boldsymbol{x}_i$$`

`$$0 = \sum^m_{i=1}\alpha_i y_i$$`

将其代入拉格朗日方程，就获得了对偶问题

`$$\begin{aligned}
\underset{\boldsymbol{\alpha}}{\max} & \sum^m_{i=1} \alpha_i - \frac{1}{2} \sum^m_{i=1} \sum^m_{j=1} \alpha_i \alpha_j y_i y_j \boldsymbol{x}_i^\mathrm{T} \boldsymbol{x}_j \\
s.t. & \sum^m_{i=1} \alpha_i y_i = 0, \\
& \alpha_i \geq 0,\quad i = 1, 2, \ldots, m.
\end{aligned}$$`

#### 核函数

令 $\mathcal{X}$ 为输入空间，$\kappa(\cdot,\cdot)$ 时定义在 $\mathcal{X} \times \mathcal{X}$ 上的对称函数，则 $\kappa$ 是核函数当且仅当对于任意数据 $D = \{\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_m\}$，“核矩阵” $\mathbf{K}$ 总是半正定的：

`$$\mathbf{K} = \begin{bmatrix}
\kappa(\mathbf{x}_1, \mathbf{x}_1) & \cdots & \kappa(\mathbf{x}_1, \mathbf{x}_j) & \cdots & \kappa(\mathbf{x}_1, \mathbf{x}_m)\\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\kappa(\mathbf{x}_i, \mathbf{x}_1) &  & \kappa(\mathbf{x}_i, \mathbf{x}_j) &  & \kappa(\mathbf{x}_i, \mathbf{x}_m)\\
\vdots & \ddots & \vdots & \ddots & \vdots \\
\kappa(\mathbf{x}_m, \mathbf{x}_1) & \cdots & \kappa(\mathbf{x}_m, \mathbf{x}_j) & \cdots & \kappa(\mathbf{x}_m, \mathbf{x}_m)
\end{bmatrix}$$`

若样本数量大于维度，适合求解原问题；

当维度高于样本数量，适合求解对偶问题。

#### 软间隔

毕竟数据不一定真正能够分开，所以引入软间隔 SVM 问题，原问题为

`$$\begin{aligned}
\underset{\boldsymbol{w}, b, \xi_i }{\min} \quad & \frac{1}{2}\left\| \boldsymbol{w} \right\|^2 + C \sum^m_{i=1} \text{loss function}\\
\text{s.t.}\quad & y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i + b) \geq 1 - \xi_i \\
&\xi \geq 0,\; i \in [m].
\end{aligned}$$`

松弛后为

`$$\begin{aligned}
\underset{\boldsymbol{w}, b, \xi_i }{\min} \quad & \frac{1}{2}\left\| \boldsymbol{w} \right\|^2 + C \sum^m_{i=1} \xi^p_i\\
\text{s.t.}\quad & y_i(\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i + b) \geq 1 - \xi_i \\
&\xi \geq 0,\; i \in [m].
\end{aligned}$$`

其中松弛变量 $\boldsymbol{\xi} = \{\xi_i\}^m_{i=1}$，$\xi_i > 0$  用以松弛**替代损失函数**，例如 hinge 损失为 `$\max (0, 1 - y_i (\boldsymbol{w}^\mathrm{T}\boldsymbol{x}_i + b))$`。西瓜书上只讲了 $p = 1$ 的情况，相当于对 $\boldsymbol{\xi}$ 使用 `$L_1$` 范数惩罚：`$\left\|\boldsymbol{\xi}\right\|_1 = \sum_i \left|\xi_i\right|$` 

#### 正则化*

#### $\epsilon$-不敏感损失函数，支持向量回归*

#### 表示定理，核方法*

## 贝叶斯分类器

#### 极大似然估计

令 $D_c$ 表示训练集 $D$ 中第 $c$ 类样本组成的集合，假设这些样本是独立同分布的，则参数 $\boldsymbol{\theta}c$ 对于数据集 $D_c$ 的似然是

`$$P(D_c \mid \boldsymbol{\theta}_c) = \prod_{\boldsymbol{x} \in D_c} P(\boldsymbol{x} \mid \boldsymbol{\theta}_c)$$ `

连乘操作容易下溢，通常使用对数似然(log-likelihood)

`$$\begin{aligned}
LL(\boldsymbol{\theta}_c) & = \log P(D_c \mid \boldsymbol{\theta}_c) \\
& = \sum_{\boldsymbol{x}\in D_c} \log P(\boldsymbol{x} \mid \boldsymbol{\theta}_c)
\end{aligned}$$`

此时参数 `$\boldsymbol{\theta}_c$` 的极大似然估计 `$\hat{\boldsymbol{\theta}_c}$` 为

`$$\hat{\boldsymbol{\theta}_c} = \underset{\boldsymbol{\theta}_c}{\arg \max} LL(\boldsymbol{\theta}_c)$$`

参数 $\boldsymbol{w}$ 的后验正比于其先验与数据似然的乘积......（参见作业4-4）

#### 朴素贝叶斯

朴素贝叶斯分类器的假设：所有属性相互独立

`$$P(c \mid \boldsymbol{x}) = \frac{P(c)P(\boldsymbol{x} \mid c)}{P(\boldsymbol{x})} = \frac{P(c)}{P(\boldsymbol{x})} \prod^d_{i=1}P(x_i \mid c)$$`

计算过程详见 hw5-1，记属性集合为 $\boldsymbol{X}$ ，结果为 $\boldsymbol{Y}$

- 计算 $\boldsymbol{Y}$ 各种取值的先验概率
- 计算各种属性取值以 $Y$ 特定取值为前提的条件概率，即 $P(X^{(i)} = i \mid Y = j)$ 的集合
- 对于每一种输入，带入上式，获取 $P(c \mid \boldsymbol{x})$ 最大者，即

`$$h_{nb}(\boldsymbol{x}) = \underset{c \in \mathcal{Y}}{\arg \max} P(c) \prod^d_{i=1} P(x_i \mid c)$$`

为了避免其他属性携带的信息被训练集中未出现的属性值“抹去”（因为连乘式的特性，未出现的属性会导致所有涉及这个属性的属性组合概率值为 0），引入**拉普拉斯修正**：

`$$\hat{P}(c) = \frac{\left| D_c \right| + 1}{\left| D \right| + N}$$`

`$$\hat{P}(x_i \mid c) = \frac{\left| D_{c,x_i} \right| + 1}{\left| D_c \right| + N_i}$$`

其中 $N$ 表示训练集 $D$ 中可能的类别数，$N_i$ 表示第 $i$ 个属性可能的取值数。感性地来看，就好像每一类的样例数都被加了一。

“拉普拉斯修正避免了因训练集样本不充分导致概率估值为零的问题，并且在训练集变大时，修正过程所引入的先验(prior)的影响也会逐渐变的可忽略，使得估值渐趋向于实际概率值。”

#### 半朴素贝叶斯分类器

**SPODE, AODE**

适当考虑一部分属性间的相互依赖信息。常用“独依赖估计”，假设每个属性在类别之外最多仅依赖于一个其他属性

`$$P(c \mid \boldsymbol{x}) \propto P(c) \prod^d_{i=1}P(x_i \mid c,pa_i)$$`

其中 $pa_i$ 为属性 $x_i$ 所依赖的属性，称为 $x_i$ 的父属性。

**TAN***

#### 高阶依赖*

#### 贝叶斯网*

#### 结构学习，推断，吉布斯采样*

### EM算法

- E步，以当前参数 $\Theta^t$ 计算隐变量 $\mathbf{Z}$ 的概率分布 $P(\mathbf{Z} \mid X, \Theta^t)$，并计算对数似然 $LL(\Theta \mid \mathbf{X}, \mathbf{Z})$ 关于 $\mathbf{Z}$ 的期望

`$$Q(\Theta \mid \Theta^t) = \mathbb{E}_{\mathbf{Z}\mid \mathbf{X},\Theta^t}LL(\Theta \mid \mathbf{X}, \mathbf{Z})$$`

- M步，寻找参数最大化期望似然。

## 聚类

#### 性能度量*

外部指标，将聚类结果与某个“参考模型”进行比较 (Jaccard 系数，FM 指数，Rand 指数)

内部指标，直接考察聚类结果而不用任何参考模型 (DB指数，Dunn指数)

#### 距离度量*

非负性、同一性、对称性、直递性

#### k-means 算法

是一个迭代算法，先随机选取 $k$ 个样本作为初始均值向量，把每个样本分类到最近的均值向量。再计算出这些聚类的均值向量，如果向量改变则更新，继续迭代直到不更新为止。

#### 学习向量量化 Learning Vector Quantization, LVQ

引入样本的类别标记和学习率 $\eta$。一开始仍然随即寻找原型向量，不断随机选取向量 $\boldsymbol{x}$，找到离它最近的原型向量。假若标记相同，则“拉近向量”：`$\boldsymbol{p}^\prime = \boldsymbol{p}_{i*} + \eta \cdot(\boldsymbol{x} - \boldsymbol{p}_{i*})$ ` 反之则“推远”向量 `$\boldsymbol{p}^\prime = \boldsymbol{p}_{i*} - \eta \cdot(\boldsymbol{x} - \boldsymbol{p}_{i*})$ ` 直至原型向量不再更新。

#### 高斯混合聚类

采用概率模型来表达聚类模型。

对 $n$ 维样本空间 $\mathcal{X}$ 中的随机向量 $\boldsymbol{x}$ ，若 $\boldsymbol{x}$ 服从高斯分布，其概率密度函数为

`$$p(\boldsymbol{x}) = \frac{1}{(2\pi)^\frac{n}{2}\left| \boldsymbol{\Sigma} \right|^\frac{1}{2}} \mathrm{e}^{\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^\mathrm{T}\boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu})}$$`

其中 $\boldsymbol{\mu}$ 是 $n$ 维均值向量，$\boldsymbol{\Sigma}$ 是 $n \times n$ 的协方差矩阵。高斯分布完全由这两个参数确定。

定义高斯混合分布

`$$p_{\mathcal{M}}(\boldsymbol{x}) = \sum^k_{i=1}\alpha_i \cdot p(\boldsymbol{x} \mid \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$`

每个混合成分对应一个高斯分布，其中 `$\boldsymbol{\mu}_i$` 和 `$\boldsymbol{\Sigma}_i$` 是第 $i$ 个高斯混合分布的参数。混合系数 `$\alpha_i > 0$`，注意所有混合系数加和为 $1$。

根据贝叶斯定理，

`$$\begin{aligned}
p_{\mathcal{M}}(z_j = i \mid \boldsymbol{x}_j) & = \frac{P(z_j = i)\cdot p_{\mathcal{M}}(\boldsymbol{x}_j \mid z_j = i)}{p_{\mathcal{M}}(\boldsymbol{x}_j)} \\
& = \frac{\alpha_i \cdot p(\boldsymbol{x}_j \mid \boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)}{\sum^k_{l=1}\alpha_l \cdot p(\boldsymbol{x}_j \mid \boldsymbol{\mu}_l,\boldsymbol{\Sigma}_l)}
\end{aligned}$$`

简记 `$\gamma_{ij} := p_{\mathcal{M}}(z_j = i \mid \boldsymbol{x}_j)$`

目标：对于每个样本 `$\boldsymbol{x}_j$` 的簇标记 `$\lambda_j$` 如下确定：

`$$\lambda_j = \underset{i \in \{1,2,\ldots,k\}}{\arg \max} \gamma_{ji}$$`

采用极大（对数）似然：

`$$\begin{aligned}
LL(D) &= \ln \left(\prod^m_{j=1} p_{\mathcal{M}} (\boldsymbol{x}_j)\right) \\
&= \sum^{m}_{j=1} \ln \left( \sum^k_{i=1} \alpha_1 \cdot p(\boldsymbol{x}_j \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_i)\right)
\end{aligned}$$`

注意由于是正态分布，所以

`$$p(\boldsymbol{x}_i \mid \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{\sqrt{2\pi\boldsymbol{\Sigma}_i}}\exp \left(-\frac{\left\|\boldsymbol{x}_i - \boldsymbol{\mu}_i\right\|^2}{2\boldsymbol{\Sigma}_i}\right)$$`

**请关注用EM求解高斯混合聚类，PPT有**

#### 密度聚类*

假设：聚类结构能通过样本分布的紧密程度确定

代表：DBSCAN

#### 层次聚类*

假设：能够产生不同粒度的聚类结果

代表：AGNES（自底向上）、DIANA（自顶向下）

## 集成学习 Ensemble Learning

序列化方法 Adaboost, 并行化方法 Bagging

Stacking

多样性问题和多样性度量