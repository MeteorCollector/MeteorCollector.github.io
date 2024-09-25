---
layout: post
title:  "jittor 踩过的坑"
date:   2024-09-24 21:52:00 +0800
categories: posts
tag: util
---

## 写在前面

最近在折腾 jittor，踩过的坑不想踩第二遍，记录于此


## 配环境

### Cuda 找不到

```
lib64/libcudart.so: cannot open shared object file: No such file or directory
```

这个的原因其实可以从 `trace` 里面看出端倪：

```
File "/path/to/your/lib/python3.8/site-packages/jittor/compiler.py", line 870, in check_cuda
    ctypes.CDLL(cuda_lib+"/libcudart.so", dlopen_flags)
```

这一段的源码如下：

```python
def check_cuda():
    if not nvcc_path:
        return
    global cc_flags, has_cuda, is_cuda, core_link_flags, cuda_dir, cuda_lib, cuda_include, cuda_home, cuda_bin
    cuda_dir = os.path.dirname(get_full_path_of_executable(nvcc_path))
    cuda_bin = cuda_dir
    cuda_home = os.path.abspath(os.path.join(cuda_dir, ".."))
    # try default nvidia-cuda-toolkit in Ubuntu 20.04
    # assert cuda_dir.endswith("bin") and "cuda" in cuda_dir.lower(), f"Wrong cuda_dir: {cuda_dir}"
    cuda_include = os.path.abspath(os.path.join(cuda_dir, "..", "include"))
    cuda_lib = os.path.abspath(os.path.join(cuda_dir, "..", "lib64"))
    if nvcc_path == "/usr/bin/nvcc":
        # this nvcc is install by package manager
        cuda_lib = "/usr/lib/x86_64-linux-gnu"
    cuda_include2 = os.path.join(jittor_path, "extern","cuda","inc")
    cc_flags += f" -DHAS_CUDA -DIS_CUDA -I\"{cuda_include}\" -I\"{cuda_include2}\" "
    if os.name == 'nt':
        cuda_lib = os.path.abspath(os.path.join(cuda_dir, "..", "lib", "x64"))
        # cc_flags += f" \"{cuda_lib}\\cudart.lib\" "
        cuda_lib_path = glob.glob(cuda_bin+"/cudart64*")[0]
        cc_flags += f" -lcudart -L\"{cuda_lib}\" -L\"{cuda_bin}\" "
        dll = ctypes.CDLL(cuda_lib_path, dlopen_flags)
        ret = dll.cudaDeviceSynchronize()
        assert ret == 0
    else:
        cc_flags += f" -lcudart -L\"{cuda_lib}\" "
        # ctypes.CDLL(cuda_lib+"/libcudart.so", import_flags)
        ctypes.CDLL(cuda_lib+"/libcudart.so", dlopen_flags)
    is_cuda = has_cuda = 1
```

在 jittor 中，CUDA 是根据 nvcc 的路径找的（在[此处](https://cg.cs.tsinghua.edu.cn/jittor/download/)亦有记载），通常是 `libnvcc` 的路径。一般来讲 `lib64` 会和 `libnvcc` 在同一文件夹下，但是在 conda 虚拟环境下只装了 `cudatoolkit` ，只有 `libnvcc`，`lib64` 并不在附近。

我的解决方式一很粗暴：直接修改 jittor 的 `compiler.py` 的源码，把正确的 `lib64` 的路径写进去。

解决方式二是用 jittor 官方给出的用 jittor 安装 cuda 的方法重新安装 cuda：

```bash
python -m jittor_utils.install_cuda
```

### 编译时 Abort

```
[i 0924 22:43:30.144030 24 jit_compiler.cc:28] Load cc_path: /home/cowa/miniconda3/envs/jitb2d/bin/g++
Aborted (core dumped)
```

这个没什么辙，怀疑是  `gcc` 或者 `g++` 的问题，换了好几个版本，重装虚拟环境并在初始指定 `gcc=9.4.0 gxx=9.4.0`，变成了 cuda 报错：

```
[i 0924 23:59:19.072964 92 jit_compiler.cc:28] Load cc_path: /home/cowa/miniconda3/envs/jitb2d/bin/g++
[i 0924 23:59:19.072992 92 jit_compiler.cc:31] Load nvcc_path: /usr/local/cuda/bin/nvcc
terminate called after throwing an instance of 'std::runtime_error'
  what():  [f 0924 23:59:19.293251 92 helper_cuda.h:128] CUDA error at /home/cowa/miniconda3/envs/jitb2d/lib/python3.8/site-packages/jittor/src/ops/array_op.cc:33  code=222( cudaErrorUnsupportedPtxVersion ) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
```

这个(应该是 cuda 版本问题)[https://github.com/Jittor/jittor/issues/194]，最后还是让 jittor 重新安装了一个 cuda。jittor 会把 cuda 安装在它的 cache 文件夹里，所以不用担心自己原来装的 cuda 被覆盖。

别忘了指定 `nvcc_path`：

```
export nvcc_path=/path/to/your/jittor/jtcuda/cuda11.2_cudnn8_linux/bin/nvcc
```

### crypt.h: No such file or directory

```shell
/home/cowa/miniconda3/envs/jitb2d_new/include/python3.8/Python.h:44:10: fatal error: crypt.h: No such file or directory
   44 | #include <crypt.h>
      |          ^~~~~~~~~
compilation terminated.
```

往虚拟环境里复制一下吧：

```shell
cp /usr/include/crypt.h /path/to/conda/envs/jitb2d_new/include/python3.8/crypt.h
```

### libgcc_s.so.1 must be installed for pthread_cancel to work

这个报错是我跑 jittor 官方示例代码跑出来的。应该情况比较多，网上一搜能搜出来一大堆。我看我虚拟环境的 `lib` 里面是有 `libgcc_s.so.1` 的，所以比较奇怪。

最后是更改 python 程序本身来解决的，要在最开头（所有 `import` 之前）进行引用：

```python
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
```

参考：[StackOverflow](https://stackoverflow.com/questions/64797838/libgcc-s-so-1-must-be-installed-for-pthread-cancel-to-work)

更改程序本身不是一个优雅的解决方案，所以如果我找到更好的解决方法，会在这里更新。

### GDB 没有 py-bt

```
#8  0x00007f876c07d5e1 in _GLOBAL__sub_I_CPUAllocator.cpp () from /home/cowa/miniconda3/envs/jitb2d/lib/python3.8/site-packages/torch/lib/libc10.so
#9  0x00007f89da9158d3 in call_init (env=0x56227b3f7940, argv=0x7fffd4831788, argc=14, l=<optimized out>) at dl-init.c:72
Undefined command: "py-bt".  Try "help".
```

## jittor 实现

#### ReLU & GeLU

jittor 中的 ReLU 和 GeLU 并没有 `inplace` 参数，但是 pytorch 里有。据说在 jittor 中 `inplace` 是默认为 `True`，和 pytorch 恰巧相反。
