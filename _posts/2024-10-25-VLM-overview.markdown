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

需要资源：调用api，需要钱

运算速度：

模型效果：

## BLIP-3-Video

是否开源：是

需要资源：没看到最低配置

运算速度：H100 (80GB显存)  1024 token: 3.3 "samples / s" , 推荐的 32 token (毕竟论文题目是 you only need 32 tokens)：8.2 "samples / s"

模型效果：

[论文链接]([xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs](https://arxiv.org/html/2410.16267v1))

关于速度的细节在 Speed 那一节。

We measure the runtime of our models in the training setting for the fair comparison. Here, we report ‘samples per second per GPU’. Without the temporal encoder (i.e., directly using 1024 visual tokens), the model processed 3.3 samples per second. With 16/32/128 tokens using the temporal encoder, the model was able to process 8.5 / 8.2 / 7.5 samples per second.

## Thoughts

模型扩容：corner case

验证方式：close-loop？但是 AD 的 VLM 要怎么 close-loop ... 对每个状态依赖有 previledged 信息的 teacher model 进行 rule-base 的标定？

其实上面两个也是 b2d 比较重要的点

novel的方式：vision -> thought chain？但是不一定每个模型都是这样想的，根据thought chain的每一步问question标定q-a对又不是很novel。

DriveLM 做过了 thought chain，是 perception -> prediction -> planning 的结构，被描述为 "full stack"