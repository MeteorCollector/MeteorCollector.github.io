---
layout: post
title:  "关于CUDA各种报错的处理"
date:   2024-04-14 16:48:00 +0800
categories: posts
tag: util
---

## 写在前面

很长时间没有写 post 了，这是因为编译原理、机器学习导论的实验和作业以及大创项目相关的工作累得让我喘不过气来再学一些东西。推免拉力赛已经开始了，我想之后会越来越忙。南大今年把夏令营提前得太早了，真是人心惶惶。

Anyway,这一条是对于我这段时间跑项目遇到过的各种奇奇怪怪报错解决方法的总结，在这里记录一下以备不时之需。

## 正文

### `arch_list[-1] += '+PTX' IndexError: list index out of range`

这是 `pytorch / extension-cpp` 项目中常见的错误。原因很有可能是没有找到可用的 CUDA 设备，这种时候还是检查一下 CUDA 吧。

[related issue](https://github.com/pytorch/extension-cpp/issues/71)

### `CUDA initialization: Unexpected error from cudaGetDeviceCount()`

```
UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW...
```

这时候一般是 `CUDA` 的库不匹配，如果输入 `nvidia-smi`，大概率会报出如下错误：

```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 535.171
```

这个问题出现的原因是kernel mod 的 Nvidia driver 的版本没有更新，一般情况下，重启机器就能够解决，如果因为某些原因不能够重启的话，也有办法reload kernel mod。详情请参照：

[很详细的blog](https://comzyh.com/blog/archives/967/)

[stackoverflow](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch)

### mm相关：`TypeError: FormatCode() got an unexpected keyword argument ‘verify‘`

用 `mm` 框架的时候报的错。这是因为环境里yapf版本过高，目前版本为 0.40.2。直接卸载重装：

```shell
pip uninstall yapf

pip install yapf==0.40.1
```

即可解决问题。

### mm安装相关

类似 pytorch, mm相关库也有自己的安装命令生成网站：

[https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html](https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html)


### c 相关

有时候与C相关的库比python相关库更容易出问题。mm里面就有大量用C写的部分。当然用C会写得更快，但是可能会消耗程序员的时间去处理版本问题。

#### `error: parameter packs not expanded with ‘...’`

这是编译的时候报的错误，和 gcc g++ 版本有关。可以手动降版本：

```shell
sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
```

#### `THC/THC.h: No such file or directory`

这是因为 `pytorch` 在 1.11 移除了 `THC/THC.h` ，所以降 `pytorch` 版本是可以解决这个问题的。