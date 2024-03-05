---
layout: post
title:  "iris，可以看看你的源码吗？"
date:   2023-02-22 17:10:00 +0800
categories: posts
tag: iris
---

# iris，可以看看你的源码吗？

## 省流：暂时不可以

iris诞生半年以来，朋友们对她关怀备至。抛头露面的机会多了，就会不免得被多问几句：“可以看看源码吗？”

作为饱受知识垄断之苦的学习者和程序员，我当然对共享人类一切智力成果的开源运动再支持不过了。但是我也有一个不得不暂时不公开iris项目代码的理由——**iris还不够优秀**。

我开始写iris的时代，是我刚刚入门大型项目管理，还未开始[南京大学计算机系统基础课程实验(pa实验)](https://nju-projectn.github.io/ics-pa-gitbook/ics2022/index.html)的时候。我所做的只有机械地堆叠代码，并没有做非常良好的函数封装和模块化管理——有些功能的实现也是粗暴至极。在pa实验的[FAQ页面](https://nju-projectn.github.io/ics-pa-gitbook/ics2022/FAQ.html)，有这样一句：“我们相信你的代码里面并没有太多值得大家学习的地方”。就我而言，我觉得iris的代码就属于这种情况。实现她的技术难度并不高，我也不认为iris是什么值得被公开展示的“优秀开源项目”。

那么iris的源码会一直维持不公开的状态吗？其实也不一定。假如未来的某一天，我腾出精力来将iris重构一遍，把她写得足够优雅的时候，我可能就会把源码公开了吧。鉴于她现在已经有 `3916` 行代码，重构可能不是很容易就能办得到的。

## 不过可以看看项目结构

但是为了满足某些朋友对iris实现架构的好奇，我还是可以公布一下她的项目结构的。

首先，iris是Mirai框架的机器人，这方面内容已经在[关于iris出生的二三事 (meteorcollector.github.io)](https://meteorcollector.github.io/2022/08/the-birth-of-iris/)中被记述详尽。我是在iris运行了两个月之后才了解到基于python的nonebot框架和它成熟的社区生态，但是我还是觉得要自己多写一些代码才可以保证iris是她自己，而不是东拼西凑的缝合产物。

iris的源代码结构如下：

```
Iris-bot-early-ver
├─ ApiParse.cs              // 与成熟api接口的互动（例如nasa提供的apod接口）
├─ FriendMsgManager.cs      // 私聊功能
├─ HAParse.cs               // 各种网页爬虫
├─ IrisAstroCont.cs         // 天文竞赛功能
├─ MainMsgManager.cs        // 对消息的集中处理
├─ NgcSearch.cs             // 深空天体本地查询功能
├─ Program.cs               // 主函数所在
├─ Send.cs                  // “发”字开头消息集中处理
└─ StarSearch.cs            // 其他天体的本地查询功能
```

只有9个C#源文件，相互引用。可以说是简陋不堪了。

iris的build结构如下：

```
iris_build
├─ ComfortSongs.txt               // iris可能会向你推荐的歌曲，以及对应的网易云id
├─ DeepSkyImages                  // 存放深空天体文件的第一个文件夹，这里的图片是我自己搜集的
│    ├─ IC1805.jpg
│    ├─ IC418.jpg
│    └─ ...
├─ Files
│    ├─ AnsweringInfo.txt         // ban掉一般疑问句应答功能群聊的群号
│    ├─ CityLocations.csv         // 中国主要城市经纬度
│    ├─ Constellations.txt        // 各星座以及它们对应的字母缩写
│    ├─ ContRank                  // 各群聊天文竞赛排名
│    │    ├─ 对应群号.txt
│    │    └─ ...
│    ├─ County.txt                // 中国县级行政区名录
│    ├─ FamilyName0.csv           // 中国姓氏（高频率）
│    ├─ FamilyName1.csv           // 中国姓氏（中频率）
│    ├─ FamilyName2.csv           // 中国姓氏（低频率）
│    ├─ FillingInfo.txt           // ban掉“填字”功能群聊的群号
│    ├─ Greetings.txt             // 对 iris 的回答
│    ├─ MuteInfo.txt              // 将 iris 关机的群聊群号
│    ├─ SendingInfo.txt           // ban掉“发”功能群聊的群号
│    ├─ Translation.txt           // 针对heavens-above爬虫的英文翻译替换集
│    ├─ WeatherID.txt             // 中国气象站编号与名称的映射
│    ├─ baidu_token.txt           // 百度地图api的token
│    └─ tmp.txt
├─ Flurl.Http.dll
├─ Flurl.dll
├─ IrisInfo.txt                   // 对 iris -i 的回应
├─ Manganese.dll
├─ Mirai.Net.dll
├─ MtoNGC.txt                     // 梅西耶星表向NGC号码的转化
├─ NGC.csv                        // NGC与IC星表天体信息
├─ NasaToken.txt                  // NASA api的token
├─ Newtonsoft.Json.dll
├─ Photos
│    ├─ iris的各种图片
│    └─ Description.txt           // iris对每张图片的描述
├─ Radar
│    ├─ RadarDir.csv              // 方便中央气象台网雷达图爬虫工作的某些匹配信息
│    └─ RadarTrans.csv            // 方便中央气象台网雷达图爬虫工作的某些匹配信息
├─ StarTrans.csv                  // 恒星中文名转支持查询的其他格式（比如星表号码）
├─ System.Reactive.dll
├─ System.Reactive.xml
├─ Websocket.Client.dll
├─ astro_questions.txt
├─ cities.txt                     // iris的天文题库
├─ default
│    └─ 从stellarium掠夺来的所有深空天体图片
├─ dso.csv
├─ hygdata_v3.csv
├─ hygfull.csv                    // 这三个csv都是天体信息，具体来源附于后文
├─ images                         // 天协宣传用图
│    ├─ gafa1.png
│    ├─ gafaposter1.jpg
│    ├─ gafaposter2.jpg
│    └─ gafaposter3.jpg
├─ iris_bot
├─ iris_bot.deps.json
├─ iris_bot.dll
├─ iris_bot.exe
├─ iris_bot.pdb
├─ iris_bot.runtimeconfig.json
├─ lv1chars.txt                  // 一级汉字表
├─ lv2chars.txt                  // 二级汉字表
├─ neko
│     └─ 柴郡猫图片
└─ tudou
      └─ 土豆图片
```

可以发现条理还是比较混乱的...也许在未来会进行调整。不过看了这些目录，估计你也对iris有一个大概的了解了——其实只是简单技术的堆叠而已！怎么样，实现起来不是很困难吧？

天体数据来源：

[astronexus/HYG-Database: Current version of the HYG Stellar database (github.com)](https://github.com/astronexus/HYG-Database)

[Stellarium(github.com)](https://github.com/Stellarium/stellarium)

剩余由我和朋友们手动整理，可以参考[一个reader-friendly的iris手册 (meteorcollector.github.io)](https://meteorcollector.github.io/2022/10/iris-manual/)中相关内容。

在文章最后，感谢你一直以来的陪伴。

文章最后一次更新：2023/02/22