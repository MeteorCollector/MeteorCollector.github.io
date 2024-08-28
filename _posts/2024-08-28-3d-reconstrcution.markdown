---
layout: post
title:  "三维重建笔记"
date:   2024-08-28 15:57:00 +0800
categories: posts
tag: autonomous driving
---

## FRP教程

## 前情提要

这篇笔记是由 2024CVPR DrivingGaussian: Composite Gaussian Splatting for Surrounding
Dynamic Autonomous Driving Scenes 伸展出去的，看到什么就记什么了。

## NeRF

NeRF，全称为“**Neural Radiance Fields**”，是一种基于神经网络的三维重建技术。它主要用于从一组二维图像中生成高质量的三维场景，特别适用于场景的渲染和建模。NeRF的主要创新在于它通过一个神经网络来表示整个三维场景，并且能够生成非常逼真的视角变化效果。

### NeRF的基本原理

NeRF通过一个深度神经网络（通常是一个多层感知器，MLP）来学习场景的辐射场。具体来说，NeRF接受一个空间位置（三维坐标）和一个视角（方向）作为输入，并输出该位置在该视角下的颜色和密度。通过对多个视角下的场景进行训练，NeRF可以推测出在新视角下的场景外观。

### 工作流程

1. **输入数据**：首先，用户提供一组拍摄自不同角度的二维图像，并且知道每张图像对应的摄像机参数（如位置和方向）。
2. **网络训练**：NeRF使用这些图像及其对应的摄像机参数来训练一个神经网络，使其能够为任意给定的空间点和视角预测颜色和密度。
3. **视图合成**：训练完成后，NeRF可以生成任意新视角下的图像，甚至可以对场景进行导航和自由探索。

## NeRF 的主要工作

### [CVPR 2023] [LocalRF] Progressively optimized local radiance fields for robust view synthesis

之前的问题：First, most existing radiance field reconstruction approaches rely on accurate pre-estimated camera poses from Structurefrom- Motion algorithms, which frequently fail on in-thewild videos. Second, using a single, global radiance field with finite representational capacity does not scale to longer trajectories in an unbounded scene.

看起来之前的工作中，相机准确位置的估计和重建工作是分开的，这个工作把两者放在了一起。论文提出了一种渐进式的方法，联合估计相机姿态和辐射场。

这篇论文的核心内容是分块，文章用了slam的思想，边运动边重建，并将场景划分为 若干个小的TensorRT块。每个块单独优化辐射场和像机pose。

解决的问题：In this setting, we are faced with two main challenges: estimating accurate camera trajectory of a long path and reconstructing the large-scale
radiance fields of scenes.

中文解析：[【分块三维重建】【slam】LocalRF：逐步优化的局部辐射场鲁棒视图合成（CVPR 2023）_progressively optimized local radiance fields for -CSDN博客](https://blog.csdn.net/qq_45752541/article/details/136326549)

### 模型结构

1. **局部辐射场（Local Radiance Fields）**：

   - 为了处理大规模无界场景，论文提出了动态分配局部辐射场的方法。这些局部辐射场被训练以覆盖视频序列中特定时间窗口内的帧。
   - 每个局部辐射场负责重建场景的一部分，并且具有自己的收缩中心，允许高分辨率空间跟随相机轨迹。

   <p><img src="{{site.url}}/images/LocalRF.png" width="80%" align="middle" /></p>

2. **TensoRF**：

   - 论文选择了TensoRF作为基础表示，因为它在质量和训练速度方面表现良好。
   - TensoRF通过一个分解的4D张量来建模场景，将3D位置映射到相应的体积密度和视角依赖的颜色。

3. **空间参数化**：

   - 论文采用了一种类似于Mip-NeRF360的收缩方法，将每个点映射到一个统一的[-2, 2]空间，以适应任意长度的相机轨迹。

### 训练方法

1. **渐进式优化**：
   - 为了提高鲁棒性，论文采用了渐进式优化方案，通过一个时间窗口逐步处理输入视频序列，并增量更新辐射场和相机姿态。
   - 这种方法确保新帧被添加到之前结构的已收敛解中，有效防止陷入较差的局部最小值。
2. **局部性引入**：
   - 通过在估计的相机姿态轨迹超出当前辐射场的未收缩空间时动态创建新的辐射场，引入了局部性。
   - 每个辐射场仅使用视频的一个子集进行监督，提高了鲁棒性。
3. **损失函数**：
   - 除了使用输入帧的颜色作为监督外，论文还添加了单目深度和相邻帧之间的光流作为损失，以提高优化稳定性。
4. **优化调度**：
   - 使用Adam优化器进行参数优化，初始学习率和损失权重根据论文中的描述进行设置。
   - 通过逐步添加监督帧和调整学习率和损失权重，实现了对整个视频序列的处理。

### [CVPR 2022] Block-NeRF: Scalable Large Scene Neural View Synthesis

这也是一个分块的 NeRF，比前者更早，尺度更大一些。

以下摘自 [NeRF神经辐射场学习笔记（八）— Block-NeRF论文创新点解读-CSDN博客](https://blog.csdn.net/weixin_44292547/article/details/126426322)

Block-NeRF是一种NeRF新的延伸，用来表示大规模环境。在渲染城市规模的场景时，将城市场景分为多个模块（Blocks），并且将NeRF也单独分配给每个block进行渲染，在预测时动态地呈现和组合这些NeRFs。这种分解（decomposition）的过程将渲染时间与场景大小解耦（decouples），即分隔开两者之间的必然联系，使渲染能够扩展到任意大的环境，并允许对环境进行逐块更新（per-block updates ofthe environment）。

针对以下目的——使NERF对不同环境条件下数月捕获的城市场景的数据具有鲁棒性，Block-NeRF的一些建设性的改进：

- 外观嵌入（appearance embeddings）和通过学习的位姿优化（learned pose refinement）：应对采集数据时的环境变化和位姿错误；（environmental changes and pose errors）
  对于单个NeRF的可控曝光（controllable exposure）：通过添加曝光条件来提供在预测时修改曝光的能力；
- 提出了一种在相邻NeRFs之间的配准方法（aligning appearance between adjacent NeRFs）：通过优化不同Block-NeRF的外观嵌入信息以匹配光照条件，并使用基于每个块神经网络到新视图的距离计算的插值权重（use interpolation weights computed based on each Block-NeRF’s distance to the novel view），使整个场景实现视觉上的配准。

### [CVPR 2022] Mega-NeRF:Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs

我们使用神经辐射场 (nerf) 从大规模的视觉捕获中构建交互式3D环境，这些捕获跨越建筑物，甚至是主要从无人机收集的多个城市街区。与单个对象场景 (传统上对nerf进行评估) 相反，我们的规模提出了多个挑战，包括 (1) 需要对具有不同照明条件的数千个图像进行建模，每个图像仅捕获场景的一小部分，(2) 令人望而却步的大模型容量，使其无法在单个GPU上进行训练，并且 (3) 快速渲染将实现交互式飞行的重大挑战。为了解决这些挑战，我们首先分析大规模场景的可见性统计信息，从而激发稀疏的网络结构，其中参数专门用于场景的不同区域。我们介绍了一种用于数据并行性的简单几何聚类算法，该算法将训练图像 (或更确切地说是像素) 划分为可以并行训练的不同NeRF子模块。我们在现有数据集 (Quad 6k和UrbanScene3D) 以及我们自己的无人机镜头上评估我们的方法，将训练速度提高3倍，将PSNR提高12%。我们还在mega-NeRF之上评估了最近的NeRF快速渲染器，并介绍了一种利用时间一致性的新颖方法。我们的技术比传统的NeRF渲染实现了40倍的加速，同时保持在0.8 db以内的PSNR质量，超过了现有快速渲染器的保真度。

#### IDEA

- 关键见解是将训练像素几何划分为与每个子模块相关的小数据分片，这对于有效的训练和高精度至关重要。
- Mega-NeRF采用了前景/背景划分，我们进一步限制了前景和采样范围。
- Mega-NeRF利用几何可见性推理来分解训练像素的集合，从而允许从遥远的相机捕获的像素仍然影响空间单元。

#### 技术解析
Mega-NeRF的核心是优化NeRF模型以提高效率和渲染速度。它采用了以下关键技术：

- 分层结构：Mega-NeRF引入了一个多级表示的层次结构，使得模型能够更有效地处理复杂的场景。这种设计允许近处的细节在低层级被精确捕捉，而远处的大规模结构则在高层级被编码。

- 稀疏采样：传统NeRF采用均匀采样，但Mega-NeRF利用动态采样策略，针对每个像素仅选择关键点进行计算，减少了不必要的计算量，提升了性能。

- 并行计算：该项目充分利用GPU的并行计算能力，优化了渲染过程，实现了高分辨率下的实时交互。

- 缓存策略：通过智能缓存最近的查询结果，Mega-NeRF能够减少重复计算，进一步加快渲染速度。

摘自 [Mega-Nerf学习笔记-CSDN博客](https://blog.csdn.net/pylittlebrat/article/details/127586291) [探索Mega-NeRF: 实时的高分辨率神经辐射场建模-CSDN博客](https://blog.csdn.net/gitblog_00093/article/details/137954395)

### [CVPR 2023] Neural Fields meet Explicit Geometric Representationsfor Inverse Rendering of Urban Scenes

这个看起来可以获得材质，然后重新布光渲染，很酷

这篇论文介绍了一种名为Fegr（Neural Fields meet Explicit Geometric Representations）的新方法，它用于从拍摄的图像中重建城市场景的几何结构和恢复场景的内在属性，如材质属性和照明信息。Fegr结合了神经辐射场（NeRF）和显式网格表示的优点，能够处理复杂的光照效果，如投射阴影和镜面高光。

主要贡献和特点包括：

1. **混合渲染管线**：Fegr使用神经场来估计初级光线（通过体积渲染），并使用从神经场重建的显式网格来模拟产生高阶光照效果（如投射阴影）的次级光线。
2. **HDR照明和材质属性建模**：Fegr能够模拟场景的HDR属性，使其适用于如重光照和虚拟物体插入等下游应用。
3. **高效的光线追踪**：通过使用优化的光线追踪库（如OptiX），Fegr能够高效地计算次级光线，实现实时渲染。
4. **端到端优化**：Fegr通过最小化观察视图的重建误差，并使用多个正则化项来约束高度欠定的问题，从而实现端到端的优化。
5. **数据集和实验**：论文使用多个城市户外数据集评估Fegr，并与现有的技术进行了比较，展示了其在不同挑战性数据集上的优势。
6. **应用示例**：论文展示了Fegr在重光照和虚拟物体插入等应用中的潜在用途，并通过用户研究验证了其生成结果的逼真度。
7. **局限性**：尽管Fegr在神经渲染方面取得了重要进展，但它也有局限性，如依赖于手动设计的先验来定义正则化项，以及目前仅限于静态场景。

总体而言，Fegr通过结合神经场的高分辨率细节和显式网格的高效渲染，为城市场景的逆向渲染提供了一种有效的解决方案。

模型结构的关键组成部分如下：

1. **神经场（Neural Field）**：
   - 神经场用于表示场景的内在属性，包括几何形状、法线向量、基础颜色（Base Color）和材质属性（包括粗糙度和金属度）。
   - 神经场由多个多层感知器（MLP）组成，分别用于预测场景中每个3D位置的Signed Distance（SD）值、法线向量、基础颜色和材质属性。
2. **显式网格（Explicit Mesh）**：
   - 从神经场的Signed Distance Field（SDF）中提取显式网格，用于高效地进行光线追踪和渲染次级光线，如投射阴影和镜面高光。
   - 使用Marching Cubes算法从SDF中提取网格。
3. **HDR环境光照（HDR Lighting）**：
   - 通过一个MLP网络表示HDR环境光照，该网络将方向向量映射到高动态范围（HDR）光照强度值。
   - HDR环境光照允许进行场景操作，如重光照和虚拟物体插入。
4. **混合延迟渲染管线（Hybrid Deferred Rendering Pipeline）**：
   - **体积渲染（Volume Rendering）**：使用神经场进行体积渲染，生成G-buffer，包含每个像素的表面法线、基础颜色、材质参数和深度信息。
   - **光线追踪（Ray Tracing）**：使用从SDF提取的显式网格，通过光线追踪计算次级光线，实现对复杂光照效果的渲染。
5. **优化方案（Optimization Scheme）**：
   - 通过最小化重建误差和多个正则化项来优化神经场景表示。
   - 使用L1重建损失来最小化输入图像和使用混合渲染器渲染的视图之间的差异。
   - 引入辅助辐射场（Auxiliary Radiance Field）和深度损失（Depth Loss）来提供几何形状的辅助监督。
   - 通过法线正则化（Normal Regularization）和光照正则化（Shading Regularization）来约束问题的欠定性。
6. **损失函数（Loss Function）**：
   - **重建损失（Reconstruction Loss）**：用于优化神经场景表示，使其能够根据输入图像重建场景。
   - **几何监督损失（Geometry Supervision Loss）**：包括辐射度损失（Radiance Loss）和深度损失（Depth Loss），用于约束几何形状。
   - **法线正则化损失（Normal Regularization Loss）\**和\**光照正则化损失（Shading Regularization Loss）**：用于确保法线向量和光照效果的一致性。
   - **天空掩码损失（Sky Mask Loss）\**和\**平滑损失（Smoothness Loss）**：用于优化场景表示的其他正则化项。
7. **实现细节（Implementation Details）**：
   - 神经场和HDR环境光照的网络结构、训练过程中的采样策略、以及用于光线追踪的OptiX库的实现细节。

### [ICPL 2023] Switch-NeRF: Learning Scene Decomposition with Mixture of Experts for Large-scale Neural Radiance Fields

这篇论文介绍了一种名为Switch-NeRF的新型神经辐射场（NeRF）方法，它通过学习场景分解来有效地重建大规模场景。Switch-NeRF的核心创新在于使用了一个可学习的门控网络（gating network），该网络将3D点分派给不同的NeRF子网络进行处理。这种方法允许端到端地学习场景分解和NeRF，而不需要人为设计的场景分解规则。

主要贡献和特点包括：

1. **端到端学习**：Switch-NeRF将场景分解和NeRF的优化过程集成到一个统一的框架中，使得网络可以通过反向传播直接学习场景分解。
2. **稀疏门控混合专家（Sparsely Gated Mixture of Experts, MoE）**：Switch-NeRF采用了MoE的设计，其中门控网络动态选择并激活最适合处理每个3D点的子网络。
3. **高效的网络设计和实现**：论文提出了一种高效的网络架构和实现策略，以确保在保持高保真度场景重建的同时，计算效率也得到优化。

模型结构：

1. **门控网络（Gating Network）**：
   - 门控网络负责为每个3D点选择最合适的NeRF子网络进行处理。它是一个可训练的神经网络，输出一组门控值，表示每个点被分派到不同子网络的概率。
   - 门控网络采用Softmax函数进行归一化，并通过Top-k操作（通常是Top-1）选择一个最有可能的子网络来处理输入点。
   - 门控网络的设计允许与NeRF子网络一起端到端训练，无需人为干预。
2. **NeRF子网络（NeRF Experts）**：
   - 每个NeRF子网络是一个深层的多层感知器（MLP），专门处理由门控网络分派的3D点。
   - 子网络输出选定点的特征向量，这些特征随后用于预测空间中的密度（σ）和颜色（c）。
   - 子网络之间相互独立，但通过门控网络的协调作用，共同参与整个场景的重建。
3. **统一的头部网络（Unified Head）**：
   - 头部网络是一个共享网络，用于整合来自不同子网络的输出，并进行最终的密度和颜色预测。
   - 它接收来自选定子网络的输出特征，结合视点方向和外观嵌入（appearance embedding），预测每个3D点的颜色和密度。
4. **容量因子和全分派（Capacity Factor and Full Dispatch）**：
   - 为了在训练中高效地分派3D点，Switch-NeRF引入了容量因子，限制每个子网络处理的样本点数量，以保持计算和内存使用的平衡。
   - 测试时，Switch-NeRF采用全分派策略，确保所有3D点都被处理，从而提高测试精度。
5. **体积渲染和损失函数（Volume Rendering and Losses）**：
   - Switch-NeRF遵循传统的NeRF体积渲染流程，通过沿相机射线采样并预测颜色和密度，合成像素的颜色。
   - 模型的训练使用渲染损失（rendering loss）作为主要的监督信号，通过比较网络渲染的图像和真实图像之间的差异来优化网络参数。
6. **辅助损失（Auxiliary Loss）**：
   - 为了解决门控网络可能偏向于某些子网络的问题，Switch-NeRF引入了辅助损失，以平衡不同子网络的训练和利用。