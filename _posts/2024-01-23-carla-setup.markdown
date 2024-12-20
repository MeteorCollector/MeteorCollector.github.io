---
layout: post
title:  "Carla环境配置"
date:   2024-01-23 11:09:00 +0800
categories: posts
tag: cv
---

<p><img src="{{site.url}}/images/carla1.png" width="80%" align="middle" /></p>

因为要收集自动驾驶的图像数据所以要用Carla模拟器。Carla也不是一个很新的项目了，配置的时候也踩过不少坑。在这里记录一下以备后用。

## transfuser

我们用的是自动采集脚本项目[transfuser](https://github.com/autonomousvision/transfuser)

项目原生的`environment.yml`中，`cudatoolkit==10.2`，我现在用的是`4060`，10.2有些太老了。

在按照原`environment.yml`配完虚拟环境之后，我向上升级了一些包。经测试该项目可以支持`cuda 11.3`，所以我进行了

```bash
pip install torch==1.11.0+cu113 torchaudio==0.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

这样环境就没问题了。

可以输入下面指令进行验证：

<p><img src="{{site.url}}/images/carla2.png" width="80%" align="middle" /></p>

但是此时打开项目有可能还是会优先使用集显而不是独显，发生段错误爆显存的情况。这个时候需要调节独显的优先级[[reference](https://blog.csdn.net/weixin_39799646/article/details/116774606?utm_source=miniapp_weixin)]。

进行（应该是需要root权限的）

```bash
prime-select nvidia
```

重启即可。

辅助工具：监测显卡使用情况（这里是每1s更新，可以根据自己情况更改）

```bash
watch -n 1 nvidia-smi
```

然后就可以愉快地采集了。

## 无图形界面模式

很显然，我们是不希望我们的电脑一天到晚开着carla进行采集的，最好把它搬到其他服务器上去，术业有专攻。但是服务器上一般是没有图形化界面的，所以需要配置Carla的无头模式。

官方文档：[carla_headless](https://carla.readthedocs.io/en/stable/carla_headless/)

## 关于高版本的问题（2024.04.18）

有可能出现类似这样的错误：

```
X Error of failed request: BadDrawable (invalid Pixmap or Window parameter)
Major opcode of failed request: 149 ()
Minor opcode of failed request: 4
Resource id in failed request: 0x3e00041
Serial number of failed request: 361
```

或者 mismatch 之类，这大概率是高版本 Carla 的 Vulkan 硬件支持出错导致的。参考 Carla 的 [issue #2232](https://github.com/carla-simulator/carla/issues/2232)：

It turns out the root problem for me was that Vulkan was using Intel graphics by default instead of my NVIDIA GPU. No idea why.

For anyone who is getting errors preceded by:

`INTEL-MESA: <Error or warning>`

Even though you have an NVIDIA GPU that is active. Run this:

```bash
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
```

Carla should then run normally with Vulkan.

```bash
./CarlaUE4.sh
```

...

The nvidia_icd.json file was missing in my case. Had to create one like this one

```bash
{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.2.131" } }
```

But now Carla 0.9.12 works. 
