---
layout: post
title:  "CV概念笔记"
date:   2024-03-05 12:48:00 +0800
categories: posts
tag: cv
---


## 选择搜索算法 Selective Search

## 残差网络 ResNet

[reference1](https://zhuanlan.zhihu.com/p/91385516)

[reference2](https://blog.csdn.net/lairongxuan/article/details/91040698)

残差网络是一个基础概念。

残差操作这一思想起源于论文《Deep Residual Learning for Image Recognition》。如果存在某个$K$层的网络$f$是当前最优的网络，那么可以构造一个更深的网络，其最后几层仅是该网络$f$第$K$层输出的恒等映射(Identity Mapping)，就可以取得与$f$一致的结果；也许$K$还不是所谓“最佳层数”，那么更深的网络就可以取得更好的结果。总而言之，与浅层网络相比，更深的网络的表现不应该更差。但是更深的网络在训练过程中的难度更大，因此作者提出了残差网络的思想。

#### 残差网络的定义

<p><img src="{{site.url}}/images/ResNET.png" width="70%" align="middle" /></p>

残差网络依旧让非线形层满足 $\mathcal{F}(x, w_h)$，然后从输入直接引入一个短连接到非线形层的输出上，使得整个映射变为

$$y = \mathcal{F}(x, w_h) + x$$

这就是残差网路的核心公式，换句话说，残差是网络搭建的一种操作，任何使用了这种操作的网络都可以称之为残差网络。

训练网络的最终目的就是找到一组权重$w^\prime$，使 $F^\prime(x,w^\prime)$ 最逼近真实的函数 $H(x)$，用式子可表示为 $F^\prime(x,w^\prime) \to H(x)$。但是现实生活中，真实的函数 $H(x)$ 往往非常复杂，想要找到一个足够逼近的函数 $F^\prime(x,w^\prime)$ 比较困难，所花费的代价也较大，作者另辟蹊径减小了工作量。

作者的思路是引入恒等映射。如图所示，网络的输入依然还是 $x$，网络在输出之前还叠加了一次输入$x$，网络结构发生了变化，$w^\prime$ 变成了 $w$，$F^\prime(x,w^\prime)$ 变成了 $F(x,w)$。训练网络的目的也发生了改变，变成了找到一组权重 $w$ 使得 $F(x,w) +x \to H(x)$，即 $F(x,w) \to H(x) - x$。如果 $F(x,w)$ 的所有权重都为0，则网络就是恒等映射。

#### 问题

##### 引入残差为何可以更好的训练？

残差的思想都是去掉相同的主体部分，从而突出微小的变化，引入残差后的映射对输出的变化更敏感。

假设：在引入残差之前，输入 $x=6$，要拟合的函数 $H(x)=6.1$，也就是说平原网络找到了一组 $w^\prime$ 使得 $F^\prime (x,w^\prime) \to H(x)$。引入残差后，输入不变还是 $x=6$，要拟合的函数 $H(x)=6.1$，变化的是 $F(x,w)+x \to H(x))=6.1$，可得 $F(x,w) \to 0.1$。

如果需拟合的函数 $H(x)$ 增大了0.1，那么对平原网络来说 $F^\prime (x,w^\prime)$ 就是从6.1变成了6.2，增大了1.6%。而对于ResNet来说，$F(x,w)$ 从0.1变成了0.2，增大了100%。很明显，**在残差网络中输出的变化对权重的调整影响更大，也就是说反向传播的梯度值更大，训练就更加容易。**

##### ResNet如何解决梯度消失问题？

我们想象这么一个ResNet，它由 $L$ 个residual block组成，即多个如上图所示的单元构成。每一个单元的输入和输出表示为 $x_l$ 和 $x_{l+1}$ 。那么我们可得如下公式：

$$y_L = h(x_L) + F(x_L, w_L) = w_sx_L + F(x_L, w_L)$$

$$x_{L+1} = f(y_L)$$

注意，有两个假设：1. $x_{L+1} = f(y_L) = y_L$，2. $w_s = I$，即$w_s$为单位矩阵（恒等映射）。所以有

$$\begin{aligned}x_L & = X_{L-1} + F(x_{L-1}, w_{L-1}) \\ 
& = x_0 +\sum^{L-1}_{i=0}F(x_i, w_i) \\
& = x_l + \sum^{L-1}_{i=l}F(x_i, w_i)
\end{aligned}$$

在反向传播过程中，令$E$为总误差，有以下求导过程：

$$\frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L}\cdot\frac{\partial x_L}{\partial x_l}\left(x_l+\sum^{L-1}_{i=l}F(x_i, w_i)\right)$$

$$=\frac{\partial E}{\partial x_L}\left(1+\sum^{L-1}_{i=l}\frac{\partial}{\partial x_l}F(x_i, w_i)\right)$$

式子的第一个因子$\frac{\partial E}{\partial x_L}$表示的损失函数到达 L 的梯度，小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有权重的层，梯度不是直接传递过来的。很显然，造成梯度消失的根本原因——梯度连乘不复存在了。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。

需要注意，上述两个假设是必不可少的，否则如上关系不再成立。论文中提到，经实践证明，令 $w_s = I$ 是最优的选择。

#### 实现摘录

```python
import torch.nn as nn
import torch
from torch.nn.init import kaiming_normal, constant

class BasicConvResBlock(nn.Module):

    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False, downsample=None):
        super(BasicConvResBlock, self).__init__()

        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut:
            out += residual

        out = self.relu(out)

        return out
```