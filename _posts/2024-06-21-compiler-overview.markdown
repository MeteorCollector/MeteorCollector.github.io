---
layout: post
title:  "编译原理复习"
date:   2024-06-21 13:14:00 +0800
categories: posts
tag: compiler
---

# 编译原理复习笔记

## 第三章 词法分析 

#### 正则表达式

掌握语法

#### 状态转换图

知道怎么画

#### DFA, NFA

正则表达式 -> NFA

NFA -> DFA (子集构造法)

DFA最小化 (把所有可区分，即输入某字符去向不同的的状态分开)：初始划分 $\{S - F, F\}$ 其中 $F$ 是终止状态

## 第四章 语法分析

三板斧：“在进行高效的语法分析之前，需要对文法做以下处理”

#### 消除二义性

二义性的消除方法没有规律可循

#### 消除左递归

消除立即左递归：

`$A \to A \alpha_1 \mid \ldots \mid A\alpha_m \mid \beta_1 \mid \ldots \mid \beta_n$`

替换为

`$A \to \beta_1 A^\prime \mid \ldots \mid \beta_n A^\prime$`

`$A^\prime \to \alpha_1 A^\prime \mid \ldots \mid \alpha_m A^\prime \mid \varepsilon$`

通用算法：

将文法的非终结符号排序为 $A_1, A_2, \ldots , A_n$

```
for i=1 to n do{
	for j=1 to i-1 do {
		將形如 A_i -> A_j y 的產生式替換爲 A_i -> b_1 y | b_2 y | ... | b_k y,
		其中 A_i -> b_1 | b_2 | ... | b_k 是以 A_i 為左部的所有產生式
	}
	消除 A_i 的立即左遞歸
}
```

#### 提取左公因子

这是因为当两个产生式具有相同前缀时无法预测，所以要提取左公因子。

对于每个非终结符号 $A$，找出它的两个或者多个可选产生式体之间的最长公共前缀

例子

```
S -> i E t S e S | i E t S | a
E -> b
```

转化为

```
S  -> i E t S S' | a
S' -> e S | \epsilon
E  -> b
```

### 自顶向下的语法分析

自顶向下分析 缺点：回溯；优点：实现简单；

预测分析技术：确定性、无回溯；FIRST FOLLOW 集计算

#### FIRST集合和FOLLOW集合

FIRST易得

FOLLOW：首先将右端结束标记 \$ 加入 FOLLOW(S) 中，按照以下两个规则不断迭代：

- 如果存在产生式 $A \to \alpha B \beta$，那么 FIRST($\beta$) 中所有非 $\varepsilon$ 的符号都加入 FOLLOW($B$) 中
- 如果存在产生式 $A \to \alpha B$ 或者 $A \to \alpha B \beta$ 包含 $\varepsilon$，那么 FOLLOW($A$) 中所有符号都加入 FOLLOW($B$) 中

例题：第四章PPT P46

#### 预测分析表构造算法

对于文法 $G$ 的每个产生式 $A \to \alpha$

- 对于 FIRST($\alpha$) 中的每个终结符号 $a$，将 $A \to \alpha$ 加入到 $M[A, a]$ 中
- 如果 $\varepsilon$ 在 FIRST($\alpha$)，那么对于 FOLLOW($A$) 中的每个符号 $b$，将 $A \to \alpha$ 也加入到 $M[A, b]$ 中

最后在所有的空白条目中填入 error

若预测分析表出现冲突，这个文法是二义的。

### 自底向上的语法分析

通用框架：移入-归约 (shift-reduce)

简单LR技术 (SLR)、LR技术 (LR)

#### 句柄

如果 $S \underset{rm}{\Rightarrow} \alpha A w \underset{rm}{\Rightarrow} \alpha \beta w$，那么紧跟 $\alpha$ 之后的 $\beta$ 就是 $A \to \beta$ 的一个句柄

在一个最右句型中，句柄右边只有终结符号

移入-归约冲突：不知道是否应该归约

归约-归约冲突：不知道按照什么产生式进行归约

#### LR语法分析技术

L 表示最左扫描，R 表示反向构造出最右推导，k 表示最多向前看 k 个符号

#### LR(0)

#### 增广文法

$G$ 的增广文法 $G^\prime$ 是在 $G$ 中增加新开始符号 $S^\prime$，并加入产生式 $S^\prime \to S$ 而得到的

#### 项集闭包

项是文法的一个产生式加上在其中某处的一个点

如果 $I$ 是文法 $G$ 的一个项集，CLOSURE($I$) 这样构造：

- 将 $I$ 中各项加入 CLOSURE($I$)
- 如果 $A \to \alpha \cdot B \beta$ 在 CLOSURE($I$) 中，而 $B \to \gamma$ 是一个产生式，且项 $B \to \cdot \gamma$ 不在 CLOSURE($I$) 中，就将该项加入其中，不断应用该规则直到没有新项可加入

#### GOTO函数

$I$ 是一个项集，$X$ 是一个文法符号，则 $GOTO(I, X)$ 定义为 $I$ 中所有形如 $[A \to \alpha \cdot X \beta]$ 的项所对应的项 $[A \to \alpha X \cdot \beta]$ 的集合的闭包

#### 构造LR(0)项集规范族

从初始项集开始，不断计算各种可能的后继，直到生成所有的项集

LR(0)自动机：第四章 PPT p81

在移入后，根据原来的栈顶状态可以知道新的状态

在规约时，根据规约产生式的右部长度弹出相应状态，也可以根据此时的栈顶状态知道新的状态 P86

#### LR分析表

P91 P94

- $[A \to \alpha \cdot a \beta ]$ 在 $I_i$ 中，且 $GOTO(I_i, a) = I_i$，则 $ACTION[i, a]$ = “移入$j$”
- $[A \to \alpha \cdot]$ 在 $I_i$ 中，那么对 $FOLLOW(A)$ 中所有 $a$ ，$ACTION[i, a]$ = “按 $A \to \alpha$ 归约”
- 如果 $[S^\prime \to S\cdot]$ 在 $I_i$ 中，那么将 $ACTION[i, \$]$ 设为“接受”
- 如果 $GOTO(I_i, A) = I_j$ ，那么在 $GOTO$ 表中，$GOTO[i, A] = j$
- 空白条目设为 error