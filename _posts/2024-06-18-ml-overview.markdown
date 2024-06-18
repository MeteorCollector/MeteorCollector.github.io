---
layout: post
title:  "机器学习导论复习"
date:   2024-06-18 14:28:00 +0800
categories: posts
tag: ml
---

## 写在前面

忙活了这么长时间本校夏令营，砸了这么多精力，又熬大页又肠胃炎，最后还是本校wl了，而且要被jyy骂菜。人生中从未有过如此屈辱的时刻。

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

TODO

## 模型评估方法

防止过拟合

- 正则化
- 数据增强
- Dropout
- Early Stopping

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

