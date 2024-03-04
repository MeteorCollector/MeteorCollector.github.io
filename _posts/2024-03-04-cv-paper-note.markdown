---
layout: post
title:  "CV论文阅读笔记"
date:   2024-03-04 23:27:00 +0800
categories: posts
---

## 写在前面

其实之前就发现搞CS的很喜欢在自己个人站里保存学习笔记。我依稀记得在我上中小学的时候还是非常喜欢记笔记的，大一的时候上数学课的时候也是。但是接触计算机课程多了以后，觉得自己写总是太浪费时间，就总是完全不记笔记或者找资料东拼西凑再总结了。

但是论文这些东西就没法参照什么资料了，要有自己的理解、按自己的习惯消化才好。又要利于以后复习，所以我打算把看论文时记下的笔记保存在这里。

另外引一个我觉得不错的笔记：[人工智能基础](https://note.tonycrane.cc/cs/ai/basic/)

嗯，感觉还是要回复一些总结整理的习惯。

从下面开始就是CV论文阅读笔记了。

## MASK R-CNN

arXiv:1703.06870v3 [cs.CV] 24 Jan 2018

Kaiming He, Georgia Gkioxari, Piotr Doll ́ar, Ross Girshick, Facebook AI Research (FAIR)

[ReadPaper链接](https://readpaper.com/pdf-annotate/note?pdfId=4557556612374470657)

R-CNN已经是成熟的技术了，这方面还是有比较好的笔记，我暂不动笔

[知乎：深度学习之目标检测的前世今生](https://zhuanlan.zhihu.com/p/32830206)

为了防止这篇文章寄了，摘录如下：

### 基础知识介绍

普通的深度学习监督算法主要是用来做**分类**，在ILSVRC（ImageNet Large Scale Visual Recognition Challenge)竞赛以及实际的应用中，还包括**目标定位**和**目标检测**等任务。其中目标定位是不仅仅要识别出来是什么物体（即分类），而且还要预测物体的位置，位置一般用边框（bounding box）标记，如图1(2)所示。而目标检测实质是多目标的定位，即要在图片中定位多个目标物体，**包括分类和定位**。

目标检测对于人类来说并不困难，通过对图片中不同颜色模块的感知很容易定位并分类出其中目标物体，但对于计算机来说，面对的是RGB像素矩阵，很难从图像中直接得到狗和猫这样的抽象概念并定位其位置，再加上有时候多个物体和杂乱的背景混杂在一起，目标检测更加困难。

### 传统的目标检测算法

这难不倒科学家们，在传统视觉领域，目标检测就是一个非常热门的研究方向，一些特定目标的检测，比如人脸检测和行人检测已经有非常成熟的技术了。普通的目标检测也有过很多的尝试，但是效果总是差强人意。

传统的目标检测一般使用滑动窗口的框架，主要包括三个步骤：

1. 利用不同尺寸的滑动窗口框住图中的某一部分作为候选区域；
2. 提取候选区域相关的视觉特征。比如人脸检测常用的Harr特征；行人检测和普通目标检测常用的HOG特征等；
3. 利用分类器进行识别，比如常用的SVM模型。

### R-CNN

R-CNN是Region-based Convolutional Neural Networks的缩写，中文翻译是基于区域的卷积神经网络，是一种结合区域提名（Region Proposal）和卷积神经网络（CNN）的目标检测方法。Ross Girshick在2013年的开山之作《Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation》奠定了这个子领域的基础，这篇论文后续版本发表在CVPR 2014，期刊版本发表在PAMI 2015。

其实在R-CNN之前已经有很多研究者尝试用Deep Learning的方法来做目标检测了，包括OverFeat，但R-CNN是第一个真正可以工业级应用的解决方案，这也和深度学习本身的发展类似，神经网络、卷积网络都不是什么新概念，但在本世纪突然真正变得可行，而一旦可行之后再迅猛发展也不足为奇了。

R-CNN这个领域目前研究非常活跃，先后出现了R-CNN、SPP-net、Fast R-CNN 、Faster R-CNN、R-FCN、YOLO、SSD等研究。Ross Girshick作为这个领域的开山鼻祖总是神一样的存在，R-CNN、Fast R-CNN、Faster R-CNN、YOLO都和他有关。这些创新的工作其实很多时候是把一些传统视觉领域的方法和深度学习结合起来了，比如选择性搜索（Selective Search)和图像金字塔（Pyramid）等。

**R-CNN：**针对区域提取做CNN的object detction。

