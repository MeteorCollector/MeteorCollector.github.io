---
layout: post
title:  "做得很满意的实验"
date:   2022-12-29 12:00:00 +0800
categories: posts
tag: project
---

# 不愿意透露姓名的实验

这里记载了我2022年秋季的数字逻辑与计算机组成实验。但是由于课程教授的要求，我不太愿意让这个网页被搜索到。

这个学期我是PA和本实验双开的，都是从计算机的最底层一路做到最上层。本实验中我自己用verilog手搓了一个cpu，然后在上面运行自己写的简易计算机系统并在上面开发程序。由于程序开发和硬软件设计都是比较自由的，所以我做本实验的过程还是比较享受的。

硬件部分仓库：[2022_vivado_projects](https://github.com/MeteorCollector/2022_vivado_projects)

软件部分仓库：[digital_design_os](https://github.com/MeteorCollector/digital_design_os)

下面内容转载至我的实验报告。

## 小组成员

由于学期初的队友已经在期中之前退课跑路，所以本学期的实验全部由我独自完成。虽然期间经历了很多辛苦，不过最后的实验结果还是比较令人满意的，学到了很多。

## 项目介绍

本项目采用10MHz的单周期CPU（在实际测试中最高支持过20MHz），搭载vga显示器、键盘、时钟三个外设，在简易计算机系统中实现了完善的命令行终端，除了验收要求内的hello、fib、time指令之外，还实现了help指令、echo指令、logo动画展示以及贪吃蛇小游戏。

展示视频南大网盘链接：[https://box.nju.edu.cn/f/e0d80365e2c54988a31e/](https://box.nju.edu.cn/f/e0d80365e2c54988a31e/)

## 硬件部分：单周期CPU 和外设接口
### 综合危机：时间都去哪了？

<p><img src="{{site.url}}/images/git_lg.png" width="60%" align="middle" /></p>

本次项目的git记录。

在上一个实验中，我已经完成了一个单周期CPU，但是由于没有采用IP核实现存储器，并且某些部分的实现并不尽如人意，所以综合实践旷日持久（最高可达2-4小时），这是完全不可接受的。工欲善其事必先利其器，我做的第一件事是完善cpu的实现。

<p><img src="{{site.url}}/images/syn_time.png" width="60%" align="middle" /></p>

早期阶段的一次综合，需要花费三个多小时.

经查，发现导致综合实践过长的罪魁祸首是手动实现的指令存储器。将其实现换为ip核后，发现程序执行错误。进行大量debug工作和资料查阅后，发现vivado的ram会默认将输出信息在寄存器中锁存一个时钟周期，导致错误。将这一默认设定取消掉之后，存储器实现成功。存储器替换后，综合时间大大缩短，时长变得可以接受。

<p><img src="{{site.url}}/images/syn_time_new.png" width="60%" align="middle" /></p>

改进之后，一次综合其实只需要几分钟。开发过程中，从综合到比特流生成的时长大概稳定在40分钟左右。

### 外设接口的实现

新cpu通过testbench之后，需要考虑的就是顶层模块的设计和外设接口的实现。

我实现的计算机外设基本沿用了实验指导中的约定，但是在键盘部分有所改动，并新增了时钟接口。

- VGA：沿用了字符显示实现的显存约定，只储存ASCII码，虽然没有实现图形化界面，但是基本可以完成软件需求。显存以0x002开头，cpu只写不读，vga只读不写。
- 键盘：键盘缓冲区以0x003开头。与实验指导不同的是，键盘只写不读，cpu既读又写。这是因为我仅仅在硬件层次维护了一个写指针，使得键盘每次在新的空白部分写入新按下的按键；然而在cpu方面，计算机系统共提供了两个函数：wait\_keyboard和get\_keyboard：其中wait\_keyboard会在软件中维护一个读指针，不停扫描键盘缓冲区，直至缓冲区出现一个键码，则返回该键码，将键盘缓冲区对应地址置0；get\_keyboard函数则每次将键盘缓冲区扫描一遍，若存在键码则返回，若不存在则返回空。
- 时钟：时钟接口存储地址以0x004开头。使用分频器实现，以毫秒为单位记录从开机开始过去的时间。时钟只写，cpu只读。

<p><img src="{{site.url}}/images/schm.png" width="60%" align="middle" /></p>

实验指导中对外设接口地址映射的规定。

硬件部分的rtl线路图如下：

<p><img src="{{site.url}}/images/rtl.png" width="60%" align="middle" /></p>

  cpu部分局部放大：

<p><img src="{{site.url}}/images/cpu_rtl.png" width="60%" align="middle" /></p>

### 忽略tcl输出导致的惨剧：仿真测试

在实现硬件之后，我将一个简易程序写入指令存储器。发现程序并不如预期一般运行。于是我将机器的时钟接到仿真时钟上，对照dump文件中的汇编代码进行仿真测试。

<p><img src="{{site.url}}/images/sim.png" width="60%" align="middle" /></p>

vivado仿真测试画面

<p><img src="{{site.url}}/images/dump.jpg" width="60%" align="middle" /></p>

dump文件，手动模拟软件运行并与vivado仿真测试中的执行情况进行比对

经查，发现ddata总线工作异常。对照verilog代码，发现这条线我并没有声明，程序自动生成了一条1bit宽度的线缆，而不是32bit，导致工作异常。其实这个错误在仿真时的报错中是可以看到的，但是当时我并没有在海量的tcl console输出中注意到这条信息，浪费了时间。经过改动，仿真测试正常。

### 最后的单步测试：人形时钟

但是通过仿真测试之后，上机测试仍然异常，程序好像没有执行一样。于是我决定使用单步执行的debug方式，将时钟接到BTNC按钮上，由我手动控制时钟信号。

<p><img src="{{site.url}}/images/manual_clock.jpg" width="60%" align="middle" /></p>

手动单步执行发现程序其实被正确执行了，我这才意识到是时钟频率的问题。我将机器主频从50MHz降至10MHz，在等待比特流文件生成的过程中，我继续按了几万下按钮确认了后续的程序执行状况依旧正常。降频之后，程序执行正常。

确认了硬件部分的正确性之后，接下来要进行的就是软件部分的开发了。

## 软件部分：简单的计算机系统

### 硬件的性能以及软件的目标

目前我们已经拥有了一个10MHz单周期cpu，一个键盘，一个只能显示ASCII字符的显示器和一个毫秒单位计时的时钟。虽然配置比较寒碜，但是足够我们运行一些比较有意思的程序了。

在实验的验收要求中，要求我们的计算机有一个简易的终端，可以解析"hello"输出"Hello World"，解析"fib n"输出斐波那契数列第n项，解析"time"输出时间。在我的设想里，我希望我的设备可以播放一段字符画动画并且运行游戏。在动画方面，我制作了一个写有NJUCS和欢迎信息的logo在屏幕上移动，这个方案不需要存储很多帧的动画画面（当然如果要存的话，现有存储器也是完全够的，但是仅仅是重复的搬砖工作，意义不大），只需要计算logo在每帧的坐标就够了。在游戏方面，我的原计划是移植问题求解课程大一下学期开发的“炸弹人”游戏，但是原项目是用C++开发，并且机制较为复杂，所以最后决定用C写一个简易的贪吃蛇游戏。

为了顺利进行软件开发，除了编写了编辑显存使用的init\_vga() putch() putstr() roll\_up()(滚屏) refresh\_screen()(将写在显存缓冲区的数据写入缓存) show\_cursor()(显示光标)函数、访问键盘输入使用的get\_keyboard() wait\_keyboard() wait\_line()(在终端中获取一行输入)函数之外，我还移植了先前在计算机系统基础课程pa3中实现的stdlib库和string库函数，方便字符串操作、输入输出处理和随机数的生成。至于时钟数据的访问，声明一个指针访问对应地址就可以了。

以下是软件部分实现的所有功能的相应介绍。

### 终端

一个中规中矩的终端。

<p><img src="{{site.url}}/images/terminal_1.jpg" width="60%" align="middle" /></p>

开机以后会直接进入计算机系统终端。该终端实现了字符输入、退格删除、回车换行、光标闪烁，若一屏写满，将向上滚屏。

### help

打印指令具体行为。

<p><img src="{{site.url}}/images/help.png" width="60%" align="middle" /></p>

如果没有参数传入，将输出支持的所有指令的帮助信息。如果传入某指令名称，输出对应指令的帮助信息。

### hello

输出"Hello World!"

<p><img src="{{site.url}}/images/hello.png" width="60%" align="middle" /></p>

### echo

返回传入的信息。

<p><img src="{{site.url}}/images/echo.png" width="60%" align="middle" /></p>

### fib

fib n 返回斐波那契数列第n项；如果后续加入参数all，将打印出第0项至第n项所有数列成员。

<p><img src="{{site.url}}/images/fib.png" width="60%" align="middle" /></p>

值得注意的是，如果传入大于47的数，将报错：输出数据超过int表示范围；如果传入小于0的数，将返回0；

### time

清屏，并在屏幕左上角显示自机器启动以来过去的时间。

<p><img src="{{site.url}}/images/timer.png" width="60%" align="middle" /></p>

 按q键退出时钟，返回终端。

### sleep

播放锁屏动画。

<p><img src="{{site.url}}/images/logo_white.JPG" width="60%" align="middle" /></p>

字符画组成的logo将在屏幕上斜向移动，遇边缘反弹。按下任意键结束动画，返回终端。如果打开rgb模式，效果如下：

<p><img src="{{site.url}}/images/logo_rgb.JPG" width="60%" align="middle" /></p>

### snake

贪吃蛇游戏。进入后，首先要求玩家键入一个数字，选择难度。最简单的难度1中，蛇每秒移动1个单位；最困难的难度5中，蛇每秒移动20个单位。蛇用wasd控制方向，随时按q退出。游戏规则不再赘述。

<p><img src="{{site.url}}/images/snake_hint.png" width="60%" align="middle" /></p>

游戏开始前的选择提醒与退出后的信息。

<p><img src="{{site.url}}/images/snake_white.JPG" width="60%" align="middle" /></p>

贪吃蛇游戏画面。

<p><img src="{{site.url}}/images/snake_rgb.JPG" width="60%" align="middle" /></p>

贪吃蛇，但是rgb。