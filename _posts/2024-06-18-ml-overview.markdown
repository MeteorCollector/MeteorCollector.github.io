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

梯度 gradient:

`$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(\mathbf{x}) \\ \vdots  \\ \frac{\partial f}{\partial x_d}(\mathbf{x}) \end{bmatrix}$$`

Hessian Matrix:

`$$\nabla^2 f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1 \partial x_1}(\mathbf{x}) & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_d}(\mathbf{x})\\
\vdots  & \ddots  & \vdots \\
\frac{\partial^2 f}{\partial x_d \partial x_1}(\mathbf{x})  & \cdots & \frac{\partial^2 f}{\partial x_d \partial x_d}(\mathbf{x})
\end{bmatrix}$$`