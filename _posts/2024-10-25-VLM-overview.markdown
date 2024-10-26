---
layout: post
title:  "VLM测评，随便写写"
date:   2024-10-25 10:00:00 +0800
categories: posts
tag: util
---

## 写在前面

出于最近忙的项目的考虑，来测评或者了解一下现在市面上的 vision-language models

## OpenAI o1

是否开源：否

需要资源：

运算速度：

模型效果：



## Thoughts

模型扩容：corner case

验证方式：close-loop？但是 AD 的 VLM 要怎么 close-loop ... 对每个状态依赖有 previledged 信息的 teacher model 进行 rule-base 的标定？

其实上面两个也是 b2d 比较重要的点

novel的方式：vision -> thought chain？但是不一定每个模型都是这样想的，根据thought chain的每一步问question标定q-a对又不是很novel。