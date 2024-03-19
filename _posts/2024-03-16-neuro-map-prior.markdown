---
layout: post
title:  "神经地图先验代码阅读"
date:   2024-03-16 23:18:00 +0800
categories: posts
tag: cv
---

# Neuro Map Prior

本篇笔记是代码精读，并不涉及理论分析。

代码仓库：[github](https://github.com/Tsinghua-MARS-Lab/neural_map_prior)

仍在施工中。

## Tree

```
.
├── docs
│   ├── data_preparation.md
│   ├── getting_started.md
│   └── installation.md
├── figs
│   ├── arch.png
│   ├── demo.png
│   ├── github-demo.gif
│   ├── map_const_comp.png
│   └── vis.png
├── LICENSE
├── mmdetection3d
├── project
│   ├── configs
│   │   ├── bevformer_30m_60m_city_split.py
│   │   ├── bevformer_30m_60m_new_split.py
│   │   ├── bevformer_30m_60m.py
│   │   ├── default_runtime.py
│   │   ├── neural_map_prior_bevformer_30m_60m_city_split.py
│   │   ├── neural_map_prior_bevformer_30m_60m_new_split.py
│   │   └── neural_map_prior_bevformer_30m_60m.py
│   ├── __init__.py
│   └── neural_map_prior
│       ├── datasets
│       │   ├── base_dataset.py
│       │   ├── evaluation
│       │   │   ├── chamfer_dist.py
│       │   │   ├── eval_dataloader.py
│       │   │   ├── hdmap_eval.py
│       │   │   ├── __init__.py
│       │   │   ├── iou.py
│       │   │   ├── precision_recall
│       │   │   │   ├── average_precision_det.py
│       │   │   │   ├── tgfg_chamfer.py
│       │   │   │   └── tgfg.py
│       │   │   ├── rasterize.py
│       │   │   ├── utils.py
│       │   │   └── vectorized_map.py
│       │   ├── __init__.py
│       │   ├── nuscences_utils
│       │   │   ├── base_nusc_dataset.py
│       │   │   ├── hdmapnet_data.py
│       │   │   ├── __init__.py
│       │   │   ├── map_api.py
│       │   │   ├── nuscene.py
│       │   │   └── utils.py
│       │   ├── nuscenes_dataset.py
│       │   └── pipelines
│       │       ├── formating.py
│       │       ├── __init__.py
│       │       ├── loading.py
│       │       └── map_transform.py
│       ├── data_utils
│       │   ├── boston_split_gen
│       │   │   ├── boston_data_split.py
│       │   │   ├── detect_trip_overlap.py
│       │   │   ├── doc.md
│       │   │   ├── order_overlapping_trips_by_timestamp.py
│       │   │   ├── rotate_iou.py
│       │   │   └── trip_overlap_val_h60_w30_thr0
│       │   │       ├── sample_poses_array.pkl
│       │   │       ├── sample_tokens.pkl
│       │   │       ├── sample_trans_array.pkl
│       │   │       ├── split_scenes.pkl
│       │   │       └── trip_overlap_val_60_30_1.pkl
│       │   └── nusc_city_infos.py
│       ├── __init__.py
│       ├── map_tiles
│       │   ├── __init__.py
│       │   ├── lane_render.py
│       │   ├── local_multi_trips.py
│       │   └── nusc_split.py
│       └── models
│           ├── hdmapnet_utils
│           │   ├── angle_diff.py
│           │   └── __init__.py
│           ├── heads
│           │   ├── bev_encoder.py
│           │   └── __init__.py
│           ├── __init__.py
│           ├── losses
│           │   ├── hdmapnet_loss.py
│           │   └── __init__.py
│           ├── mapers
│           │   ├── base_mapper.py
│           │   ├── __init__.py
│           │   ├── loss_utils.py
│           │   ├── map_global_memory.py
│           │   ├── original_hdmapnet_baseline.py
│           │   ├── original_hdmapnet_nmp_final.py
│           │   ├── original_hdmapnet.py
│           │   └── set_epoch_info_hook.py
│           ├── modules
│           │   ├── custom_base_transformer_layer.py
│           │   ├── decoder.py
│           │   ├── deformable_transformer.py
│           │   ├── encoder.py
│           │   ├── gru_fusion.py
│           │   ├── __init__.py
│           │   ├── multi_scale_deformable_attn_function.py
│           │   ├── prior_cross_attention.py
│           │   ├── spatial_cross_attention.py
│           │   ├── temporal_self_attention.py
│           │   ├── transformer.py
│           │   ├── utils.py
│           │   └── window_cross_attention.py
│           └── view_transformation
│               ├── bevformer.py
│               ├── hdmapnet.py
│               ├── homography.py
│               ├── __init__.py
│               └── lss.py
├── README.md
├── requirements.txt
└── tools
    ├── create_data.py
    ├── data_converter
    │   └── nuscenes_converter.py
    ├── data_sampler.py
    ├── dist_test.sh
    ├── dist_train.sh
    ├── mmdet_dataloader.py
    ├── mmdet_train.py
    ├── test.py
    └── train.py


24 directories, 103 files
```

## docs

### data_preparation.md

这个是关于nuScenes数据集下载的文档。dataset被设在 `neural_map_prior/data/nuscenes`，下载之后，数据集结构为

```
neural_map_prior
├── mmdet3d
├── tools
├── projects
│   ├── nmp
│   ├── configs
├── ckpts
├── data
│   ├── nuscenes
│   │   ├── maps <-- used
│   │   ├── samples <-- key frames
│   │   ├── sweeps  <-- frames without annotation
│   │   ├── v1.0-mini <-- metadata and annotations
│   │   ├── v1.0-test <-- metadata
|   |   ├── v1.0-trainval <-- metadata and annotations
│   │   ├── nuScences_map_trainval_infos_train.pkl <-- train annotations
│   │   ├── nuScences_map_trainval_infos_val.pkl <-- val annotations
│   ├── nuscenes_infos
│   │   ├── train_city_infos.pkl
│   │   ├── val_city_infos.pkl
```

## figs

Nothing to comment.

## mmdetection3d

Embedded git repository: [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/99ad831a8b04a7f5622c366d8e60745f92a62008)

## tools

如果纯粹地训练/使用该模型，只会直接运行该文件夹下的脚本。这些脚本针对不同模型都是普适的，只需要更改参数就可以运行不同的模型。

```
.
├── create_data.py
├── data_converter
│   └── nuscenes_converter.py
├── data_sampler.py
├── dist_test.sh
├── dist_train.sh
├── mmdet_dataloader.py
├── mmdet_train.py
├── test.py
├── train.py
└── tree.txt

1 directory, 10 files
```

### dist_test.sh

Getting started: evaluation test 运行的脚本。实际上调用了 `tools/test.py`，完整指令为

```
./tools/dist_test.sh ./project/configs/bevformer_30m_60m.py ./ckpts/bevformer_epoch_24.pth 8 --eval iou
```

### test.py

```python
description='MMDet test (and eval) a model'
```

和他描述的一样，是一个训练/评估模型的较为普适的脚本，从CONFIG文件读取配置信息。

### train.py

同样地，是一个普适性的模型训练脚本。


## project

```
.
├── configs
│   ├── bevformer_30m_60m_city_split.py
│   ├── bevformer_30m_60m_new_split.py
│   ├── bevformer_30m_60m.py
│   ├── default_runtime.py
│   ├── neural_map_prior_bevformer_30m_60m_city_split.py
│   ├── neural_map_prior_bevformer_30m_60m_new_split.py
│   └── neural_map_prior_bevformer_30m_60m.py
├── __init__.py
├── neural_map_prior
│   ├── datasets
│   │   ├── base_dataset.py
│   │   ├── evaluation
│   │   │   ├── chamfer_dist.py
│   │   │   ├── eval_dataloader.py
│   │   │   ├── hdmap_eval.py
│   │   │   ├── __init__.py
│   │   │   ├── iou.py
│   │   │   ├── precision_recall
│   │   │   │   ├── average_precision_det.py
│   │   │   │   ├── tgfg_chamfer.py
│   │   │   │   └── tgfg.py
│   │   │   ├── rasterize.py
│   │   │   ├── utils.py
│   │   │   └── vectorized_map.py
│   │   ├── __init__.py
│   │   ├── nuscences_utils
│   │   │   ├── base_nusc_dataset.py
│   │   │   ├── hdmapnet_data.py
│   │   │   ├── __init__.py
│   │   │   ├── map_api.py
│   │   │   ├── nuscene.py
│   │   │   └── utils.py
│   │   ├── nuscenes_dataset.py
│   │   └── pipelines
│   │       ├── formating.py
│   │       ├── __init__.py
│   │       ├── loading.py
│   │       └── map_transform.py
│   ├── data_utils
│   │   ├── boston_split_gen
│   │   │   ├── boston_data_split.py
│   │   │   ├── detect_trip_overlap.py
│   │   │   ├── doc.md
│   │   │   ├── order_overlapping_trips_by_timestamp.py
│   │   │   ├── rotate_iou.py
│   │   │   └── trip_overlap_val_h60_w30_thr0
│   │   │       ├── sample_poses_array.pkl
│   │   │       ├── sample_tokens.pkl
│   │   │       ├── sample_trans_array.pkl
│   │   │       ├── split_scenes.pkl
│   │   │       └── trip_overlap_val_60_30_1.pkl
│   │   └── nusc_city_infos.py
│   ├── __init__.py
│   ├── map_tiles
│   │   ├── __init__.py
│   │   ├── lane_render.py
│   │   ├── local_multi_trips.py
│   │   └── nusc_split.py
│   └── models
│       ├── hdmapnet_utils
│       │   ├── angle_diff.py
│       │   └── __init__.py
│       ├── heads
│       │   ├── bev_encoder.py
│       │   └── __init__.py
│       ├── __init__.py
│       ├── losses
│       │   ├── hdmapnet_loss.py
│       │   └── __init__.py
│       ├── mapers
│       │   ├── base_mapper.py
│       │   ├── __init__.py
│       │   ├── loss_utils.py
│       │   ├── map_global_memory.py
│       │   ├── original_hdmapnet_baseline.py
│       │   ├── original_hdmapnet_nmp_final.py
│       │   ├── original_hdmapnet.py
│       │   └── set_epoch_info_hook.py
│       ├── modules
│       │   ├── custom_base_transformer_layer.py
│       │   ├── decoder.py
│       │   ├── deformable_transformer.py
│       │   ├── encoder.py
│       │   ├── gru_fusion.py
│       │   ├── __init__.py
│       │   ├── multi_scale_deformable_attn_function.py
│       │   ├── prior_cross_attention.py
│       │   ├── spatial_cross_attention.py
│       │   ├── temporal_self_attention.py
│       │   ├── transformer.py
│       │   ├── utils.py
│       │   └── window_cross_attention.py
│       └── view_transformation
│           ├── bevformer.py
│           ├── hdmapnet.py
│           ├── homography.py
│           ├── __init__.py
│           └── lss.py
└── tree.txt

18 directories, 82 files
```

### configs

这里存放的都是模型的`config`文件。训练/测试不同的文件只需要在运行 `tools/test` 或者 `tools/train` 时传入不同的文件作为 `config` 信息就可以了。

在 `plugin_dir = 'project/neural_map_prior/'` 这一行中，以 `plugin_dir` 参数传入了 `neuro_map_prior`

#### bevformer_30m_60m.py

`BEVformer map baseline with resnet101 backbone`

Getting started: evaluation test调用的默认模型`config`文件

### __init__.py

### neural_map_prior

#### __init__.py

```python
from .models import *
from .datasets import *
```

#### ckpts

需要用户自己创建一个 `ckpts` 文件夹，在这里存储训练获得的 `checkpoint` 。

#### datasets

- **__init__.py**

  ```python
  from .pipelines import *
  from .nuscenes_dataset import *
  from .base_dataset import *
  
  __all__ = ['nuScenesMapDataset', 'BaseMapDataset']
  ```

- **base_dataset.py**

  This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI dataset.

  这里定义了为了加载主流地图数据写的dataset的基类 `BaseMapDataset`。重载了 `pytorch` 中 `dataset` 的若干函数并添加了一些自定义函数。

- **nuscenes_dataset.py**

  前半部分主要是 `config` 信息，有 `img_norm_cfg`, `eval_cfg`, `train_pipeline`, `MAPS`

  后半部分以 `BaseMapDataset` 为基类定义了 `nuScenesMapDataset`。

- **evaluation**

  - **chamfer_dist.py**

    一个计算倒角距离的脚本，主要是一些辅助功能。

  - **eval_dataloader.py**

    定义了 `HDMapNetEvalDataset(object)`     

  - **hdmap_eval.py**

    `Evaluate nuScenes local HD Map Construction Results`

    和描述的一样，这里主要 `evaluate` 地图重建结果。这里主要调用了很多 `IoU` 而不是 `Chamfer Distance`，说明生成的仍然是 `bounding box` 而不是三维点云？

  - **__init__.py**

    没有任何内容。

  - **iou.py**

    计算 `IoU` 的辅助函数。

    `get_batch_iou` 计算两个地图之间的IoU， 该函数首先将地图转换为布尔类型，然后逐通道计算交集和并集，并将结果返回（不相除，分别返回这一batch的intersection和union列表）。

    `get_batch_iou_bound` 计算的 IoU 在 predict map 和 ground truth 的边界范围内（返回类型和上一个函数一样）。

    `get_batched_iou_no_reduction` 返回的是将 intersection 和 union 相除后得到的数值。

  - **precision_recall**

    - **average_precision_det.py**

      这里面东西比较杂，主要是算准确度的。

      首先是 `average_precision(recalls, precisions, mode='area')`，这是一个单纯的计算平均准确率的函数，当 `mode` 为 'area' 时，计算 Precision-Recall (PR) 曲线下的面积作为平均精度；当 `mode` 为 '11points' 时，计算召回率在 11 个点（0.0 到 1.0 之间，步长为 0.1）处的精度并求平均。返回值为计算得到的平均精度。

      `tpfp_test`: 这个函数用于计算给定的检测结果（det, detection）与真实标注（gt, ground truth）之间的真正例（true positives）和假正例（false positives）。它接受检测结果det_bboxes和真实标注gt_bboxes，以及阈值threshold，然后，根据置信度阈值将检测结果分为真正例和假正例，并返回它们的数量。

      `get_cls_results`: 这个函数用于从给定的检测结果和真实标注中获取特定类别的检测结果和标注信息。它接受检测结果det_results、标注信息annotations和指定的类别IDclass_id。然后，根据class_id筛选出特定类别的检测结果和标注信息，并返回`cls_dets`, `cls_gts`, `cls_gts_mask`。

      `_eval_map(det_results, annotations, threshold=0.5, num_classes=3, class_name=None, logger=None, tpfp_fn_name='vec', nproc=4)`：

      ```
      det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
      The outer list indicates images, and the inner list indicates per-class detected bboxes.
      annotations (list[dict]): Ground truth annotations where each item of he list indicates an image. Keys of annotations are:

        - `bboxes`: numpy array of shape (n, 4)
        - `labels`: numpy array of shape (n, )

      scale_ranges (list[tuple] | None): canvas_size
      Default: None.

      iou_thr (float): IoU threshold to be considered as matched.
      Default: 0.5.

      logger (logging.Logger | str | None): The way to print the mAP summary. See `mmcv.utils.print_log()` for details. 
      Default: None.
      
      tpfp_fn (callable | None): The function used to determine true/false positives. If None, :func:`tpfp_default` is used as default unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this case). If it is given as a function, then this function is used to evaluate tp & fp. Default None.
      
      nproc (int): Processes used for computing TP and FP. Default: 4.

      Returns:
      tuple: (mAP, [dict, dict, ...])
      ```

      `eval_map(cfg: dict, logger=None)`：是 `_eval_map` 外面套的一层，负责把 `cfg` 里的参数读出来传进去

      `print_map_summary`：没什么好说的，就是打印信息

    - **tgtg_chamfer.py**

      评估模型用的辅助函数。

      `vec_iou`：用于计算 `pred_lines` 和 `gt_lines` 之间的IoU；

      `convex_iou`：用于计算一个预测凸多边形与一个真实多边形之间的IoU；

      `rbbox_iou`：用于计算一个预测旋转矩形与一个真实多边形之间的交并比；

      `polyline_score`：用于计算两个多边形之间的评估分数，然后通过计算最近点之间的欧几里得距离来确定多边形的相似度，也就是chamfer距离。

    - **tgtg.py**

      这里的函数是调用 **tgtgchamfer.py** 来计算 `tp` 和 `fp` 。其实一直想吐槽的是，这个文件名真的没写错吗？总感觉g要改成p才对。

      `tpfp_bbox` 和 `tpfp_rbbox` 用于检查检测到的边界框是否为（true positive）还是误检的正样本（false positive）。

      `tpfp_det` 的描述是 `Check if detected bboxes are true positive or false positive.`

      传入参数：

      `det_bboxes`：检测到的边界框，形状为 $(m, 5)$，其中 $m$ 是边界框的数量，每个边界框由 $4$ 个坐标和一个置信度组成。

      `gt_bboxes`：真实的边界框，形状为 $(n, 4)$，其中 $n$ 是真实边界框的数量。

      `threshold`：IoU 阈值，用于确定真正例和误检例，默认为 $0.5$。

      然后返回 `tp` 和 `fp`，分别是两个数组，初始化为全$0$，分别在 `tp` 和 `fp` 对应 `id` 处置 $1$。

      `tpfp_gen` 在源文件里的描述是 `tpfp_det` 内容的完全复制，感觉是搞错了。但是感觉确实只是上面的 `det_bboxes` 参数换成了 `gen_lines`，其他部分完全一致。

  - **rasterize.py**

    这个脚本做的事情是把矢量地图栅格化。

    `get_patch_coord(self, patch_box, patch_angle=0.0)` 给定一个 patch 的框架信息，返回 patch 的坐标。第三个参数是指定角度进行旋转。

    `mask_for_lines` 画栅格化的线条。

    `line_geom_to_mask(self, geom, map_mask, thickness, color)` 将矢量线条集合对象 `geom` 转换为栅格化的线条集合

    `preprocess_data(self, vectors)` 预处理矢量数据。根据 patch 的大小和画布的大小对矢量数据进行缩放和平移，然后返回处理后的数据字典。

    `rasterize_map(self, vectors, thickness)` 根据矢量生成栅格化地图。

  - **utils.py**

    辅助函数。

    一个单独的 `get_pos_idx`，输入pos_msk: $b \times k$，输出 remain_idx: $b \times n$ remain_idx_mask: $b \times n$

    然后单独定义了 `CaseLogger` 类，用于记录、找到典型 case 并保存等等。

  - **vectorized_map.py**

    定义了 `class VectorizedLocalMap(object)` 和一系列关于矢量地图的方法。

    `gen_vectorized_samples(...)`: 用于生成矢量化的局部地图数据。它接受位置信息、车辆到全局坐标系的平移和旋转信息，并返回过滤后的矢量数据。在该方法中，通过调用 `get_map_geom` 方法获取指定要素的几何信息(`shapely.geometry`)，然后通过一系列方法将几何信息转换为矢量数据。最终将所有有效的矢量数据存储在 `filtered_vectors` 中并返回。

    `get_map_geom(...)`: 这个方法用于获取指定范围内、指定 `layer` 的几何信息。它接受 `patch` 的尺寸和角度、`layer` 的类别和问询的位置，并返回符合条件目标的几何信息。具体是通过调用 `NuScenesMapExplorer` 来实现的。

    `_one_type_line_geom_to_vectors(...)`: 这是一个辅助方法，用于将特定类型的线条几何转换为矢量。它接受线条几何信息，并返回矢量化的线条数据。

    `poly_geoms_to_vectors(...)`, `line_geoms_to_vectors(...)`, `ped_geoms_to_vectors(...)`: 这三个方法分别用于将多边形、线条和人行横道几何转换为矢量数据。它们通过调用 `_one_type_line_geom_to_vectors` 方法实现。

    `get_ped_crossing_line(...)`: 这个方法用于获取人行横道的线条信息。它通过解析地图中的人行横道多边形信息来获取相应的线条，并返回这些线条的列表。

    `sample_pts_from_line(...)`: 这个方法用于从线条中抽样点。它接受线条几何信息，并返回采样后的点坐标以及有效点的数量。


- **nuscenes_utils**

  - **base_nusc_dataset.py**

    定义了 `class NuscData(torch.utils.data.Dataset)` 类，还是老几样，以 pytorch 的 Dataset 为基类。

  - **hdmapnet.py**

    把上一个文件中的 `NuscData` 作为基类，定义 `class RaseterizedData(NuscData)` （栅格化数据）。

    同时还有一个 `gen_topdown_mask` 函数，生成鸟瞰视图的掩码。

  - **__init__.py**

    这里面什么也没有

  - **nuscene.py**

    定义了 `MyNuScenesMap(NuScenesMap)` 和 `MyNuScenesMapExplorer(NuScenesMapExplorer)` 两个类。

    `MyNuScenesMap` 里的方法只有 `get_map_mask`，描述是`Return list of map mask layers of the specified patch`，返回格式是 `Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas`。

    `MyNuScenesMapExplorer` 里的方法比较多，提供了各种几何图形类型（线段、多面体）往几何图形数据类型/掩码（本身的 `Numpy ndarray line/polygon mask` 或者在地图上鸟瞰的 $h \times w$ 二维掩码（需要指定生成的局部范围）。

    单独定义了两个函数：

    `def gen_topdown_mask(nuscene, nusc_maps, sample_record, patch_size, canvas_size, seg_layers, thickness=5, type='index', angle_class=36)` 生成鸟瞰掩码

    `extract_contour(topdown_seg_mask, canvas_size, thickness=5, type='index', angle_class=36)` 生成轮廓

  - **map_api.py**

    定义了 `CNuScenesMapExplorer(NuScenesMapExplorer)`，但是初始化函数写成了 `__ini__`，这个类真的有在用吗？

    然后额外定义了一个 `plot` 函数，看起来是把一系列 `Polygon` 画在图上。

    很混乱，这个文件。

  - **utils.py**

    这里都是关于地图的工具函数。

    `get_rot(h)`：用于将向量绕原点旋转角度 h；

    `plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx, alpha_poly=0.6, alpha_line=1.)`：绘制 NuScenes 地图的可视化效果。它从 NuScenes 地图中提取指定范围内的地图信息，并绘制不同类型的道路、车道、行人穿越等元素。参数 rec 是 NuScenes 的记录，nusc_maps 是地图数据，nusc 是 NuScenes 数据集对象，scene2map 是场景到地图的映射，dx 和 bx 是某种比例系数，alpha_poly 和 alpha_line 是绘制多边形和线条的透明度。

    `get_discrete_degree(vec, angle_class=36)`：根据给定向量的角度，返回离散角度。参数 vec 是一个二维向量，angle_class 是角度分段数。

    `get_local_map(nmap, center, stretch, layer_names, line_names)`：从地图中提取局部地图信息。该函数从地图中获取指定范围内的多边形和线条，并将它们转换为局部坐标系。参数 nmap 是地图对象，center 是局部坐标系的中心，stretch 是拉伸系数，layer_names 和 line_names 是地图中的多边形和线条的名称。

    `get_lidar_data(nusc, sample_rec, nsweeps, min_distance)`：从 NuScenes 数据集中获取 LiDAR 数据。该函数返回一定数量的 LiDAR 扫描的点云数据。参数 nusc 是 NuScenes 数据集对象，sample_rec 是样本记录，nsweeps 是扫描的数量，min_distance 是距离阈值，用于去除距离太近的点。

    `img_transform(img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate)`：对图像进行变换，包括调整大小、裁剪、翻转和旋转。此外，该函数还返回变换后的旋转矩阵和平移向量。

    `NormalizeInverse(mean, std)`：标准化的逆操作，用于将归一化后的图像恢复为原始图像。这个类在调用时会将张量进行复制，以避免修改原始张量。

    `gen_dx_bx(xbound, ybound, zbound)`：生成偏移量和中心点，用于计算局部地图。

    `get_nusc_maps(map_folder)`：从文件夹中加载 NuScenes 地图数据。

    `label_onehot_encoding(label, num_classes=4)`：将标签转换为 one-hot 编码。

    `pad_or_trim_to_np(x, shape, pad_val=0)`：将数组填充或裁剪到指定形状。

- **pipelines**

  这个文件夹是关于用于数据处理的自定义数据管道（pipeline）。用了很多`mmcv`库函数。所有类都注册到了 `mmdet.datasets.builder.PIPELINES` ,可以在数据加载时调用。这些自定义数据管道可以根据具体的数据和任务进行灵活配置，以实现不同的数据预处理和增强操作。

  - **__init.py__**

    ```python
    from .loading import LoadMultiViewImagesFromFiles
    from .map_transform import VectorizeLocalMap
    from .formating import FormatBundleMap, Normalize3D, Pad3D, ResizeCameraImage

    __all__ = [
        'LoadMultiViewImagesFromFiles',
        'VectorizeLocalMap',
        'FormatBundleMap',
        'Normalize3D',
        'Pad3D',
        'ResizeCameraImage'
    ]
    ```

  - **formating.py**

    定义了类 `FormatBundleMap(object)`

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer

    定义了 `Normalize3D` 类用于对图像进行归一化处理；

    定义了 `Pad3D` 类用于对图像和掩码进行填充。它可以根据指定的大小或者大小的倍数对图像和掩码进行填充。

    `ResizeMultiViewImages` 、`ResizeCameraImage` 调整多视图/摄像头图像的大小。

  - **loading.py**

    `class LoadMultiViewImagesFromFiles(object)`:

    Load multi channel images from a list of separate channel files.

  - **map_transform.py**

    定义了 `VectorizeLocalMap` 类，提供了非常多的关于向量化的函数。

#### data_utils

  - **boston_split_gen**

    
    - **doc.md**

      Sequentially run the following commands to produce the Boston split in the paper.

      1. `python detect_trip_overlap.py`
      2. `python order_overlapping_trips_by_timestamp.py`
      3. `python boston_data_split.py`

    - **detect_trip_overlap.py**

      这个文件比较**重要**。

      文件中的函数用来处理 NuScenes 数据集的相关操作，包括获取样本标记、样本姿态、转换记录、地图角度以及过滤场景等功能。

      `get_sample_token`: 从给定的 NuScenes 数据集和场景中获取样本标记列表。样本标记是按照时间顺序组织的，通过首个样本标记的下一个标记来获取整个样本序列的标记列表。

      `get_trans_from_record`: 根据给定的车辆姿态记录，获取转换矩阵和姿态信息。将姿态信息转换为四元数（Quaternion），并计算转换矩阵，以及姿态信息（位置和旋转角度）。

      `get_map_angle`: 根据车辆姿态记录中的全局旋转角度，计算地图角度。使用四元数表示姿态，将其转换为欧拉角，并将角度值转换为以度为单位的角度。

      `get_sample_pose`: 根据给定的 NuScenes 数据集和样本标记列表，获取样本的姿态信息。遍历样本标记列表，获取每个样本对应的激光雷达数据和车辆姿态记录，然后调用 `get_trans_from_record` 函数获取姿态信息。

      `filter_scenes`: 从给定的 NuScenes 数据集和场景列表中过滤出符合条件的场景。根据场景名称和地理位置信息，判断是否属于指定的场景，并返回符合条件的场景列表。

      最核心的函数是 `find_hdmap_history(args)`。这段代码的功能是根据输入参数从 NuScenes 数据集中查找历史高清地图（HDMap）数据。

      使用 NuScenes 类初始化数据集对象 nusc，并根据参数设置 verbose 为 True。

      调用 create_splits_scenes() 函数创建划分的场景，并通过 filter_scenes() 函数过滤出与指定 hist_set 匹配的场景。

      获取划分场景的地理位置信息 split_scenes_loc 和样本标记信息 sample_tokens。

      根据样本标记信息，通过调用 get_sample_pose() 函数获取样本的位置姿态信息 sample_poses_trans，将其转换为数组形式 sample_poses_array 和样本平移向量 sample_trans_array。

      创建保存目录 args.save_dir_name（默认为`trip_overlap_val_h60_w30_thr0`），并将 `split_scenes`、`sample_tokens`、`sample_poses_array` 和 `sample_trans_array` 分别保存为 pkl 文件。（**下面很快就会用到**）

      设置 HDMap 的高度 hdmap_height 和宽度 hdmap_width。

      初始化 hdmap_history_dict 字典，用于存储历史 HDMap 数据。

      遍历样本标记信息，对每个样本计算与其他样本的距离，并检查是否满足条件生成历史 HDMap 数据。

      根据条件判断计算两个 HDMap 区域的交并比（IoU），并根据给定的阈值 args.iou_thr 进行筛选。

      将符合条件的历史 HDMap 数据保存到 hdmap_history_dict 字典中，并将其保存为 pkl 文件。
      
    - **order_overlapping_trips_by_timestamp.py**

      这一段主要是对重叠行程数据进行处理，生成了 token 到遍历 ID 的映射，并将结果保存到文件中供后续使用。
    
      1. 加载样本标记序列（从`./trip_overlap_val_h60_w30_thr0/sample_tokens.pkl`加载）和重叠行程数据。

      2. 通过遍历重叠行程数据（从`./trip_overlap_val_h60_w30_thr0/trip_overlap_val_60_30_1.pkl`加载），生成了 token 到重复序列和遍历 ID 的映射。

      3. 根据加载的 NuScenes 数据集，计算了重复序列中每个样本的时间戳，并按时间对重复序列进行排序。

      4. 更新 token 到遍历 ID 的映射，并将结果保存到文件（`val_token2traversal_id.pkl`）中。

    - **boston_data_split.py**

      这段代码是用于加载数据并根据城市名和地理位置对数据进行过滤和划分。

      *不过这里的很多路径设定都是硬编码作者自己电脑里的路径，也没有相关文档，可能不太需要再去用这些脚本手动处理这些数据了吧*

    - **rotate_iou.py**

      这里主要是一些手动实现的运算函数，把cpu运算搬到了gpu，狠狠加速。引用自[这里](https://github.com/hongzhenwang/RRPN-revise)。


    - **nusc_city_infos.py**

      把 `/public/MARS/datasets/nuScenes` 里的数据进行整理，存成 `train_city_infos.pkl` 和 `val_city_infos.pkl`。

    - **trip_overlap_val_h60_w30_thr0**

      上文代码生成的 `pkl` 都在这里了。

      - **samples_poses_array.pkl**

      - **samples_tokens.pkl**

      - **samples_poses_array.pkl**

      - **samples_trans_array.pkl**

      - **split_scenes.pkl**

      - **trip_overlap_val_60_30_1.pkl**

  

#### map_tiles

  - **__init__.py**

    什么也没有。

  - **lane_render.py**

    上面是一大堆辅助函数，是为了实现`draw_global_map(args)`：

    首先配置地图参数，包括保存路径、地图属性等信息；然后创建全局地图切片字典，用于存储地图切片的数据；加载数据集的信息，包括样本的城市名称、地图的最小和最大地理位置等；根据地图切片的大小和城市名称，获取样本 token 到地图索引的映射；遍历数据集中的样本信息，依次处理每个样本对应的地图切片；根据样本的城市名称和地图索引，获取地图切片的边界坐标；从结果根目录中加载处理后的 BEV 特征，将其投影到**全局地图切片**中；更新地图切片字典中的数据；保存处理后的**全局地图切片**到指定路径（默认是当前位置）；将保存的地图切片可视化，并保存为图像文件。

  - **local_multi_trips.py**

    主要还是处理了`hdmap_history_loc_val_h60_w30_thr0/hdmap_history_{dataset}_30_60_1`，主要流程如下：

    设置结果根目录和保存根目录，并定义 BEV（Bird's Eye View）属性，包括真实高度、真实宽度、BEV 高度和宽度。

    加载历史轨迹信息，包括样本 token、样本姿态数组、样本变换数组和 HDMap 历史字典。

    加载验证集数据信息，包括数据信息和数据城市名称。

    统计每个样本的重叠轨迹数目，绘制直方图并保存。

    对重叠轨迹数目进行排序，遍历每个样本，生成全局地图的可视化效果并保存到文件中。

  - **nusc_split.py**

    根据 nuScenes 数据集的配置，提取训练集和验证集中所有样本的 token，并保存到相应的 pickle 文件中。

    根据预定义的训练集和验证集场景列表，过滤出对应的场景信息；遍历训练集和验证集的场景信息，分别获取每个场景中的样本 token 列表；将所有样本 token 列表合并成一个列表，分别保存为训练集和验证集的样本 token pickle 文件；打印训练集和验证集中样本的数量。

    主要目的是为后续的处理和训练准备数据，提供了训练集和验证集中所有样本的 token 列表。

    存储路径：

    ```python
    with open('train_sample_tokens.pkl', 'wb') as f:
        pickle.dump(sample_tokens_train, f)
    with open('val_sample_tokens.pkl', 'wb') as f:
        pickle.dump(sample_tokens_val, f)
    ```


#### models

  - **__init.py__**

    ```python
    from .heads import *
    from .losses import *
    from .mapers import *
    from .modules import *
    from .view_transformation import *
    ```

  - **hdmapnet_utils**

    - **angle_diff.py**

      这段代码是用于计算预测角度和真实角度之间的差异。

      函数 `onehot_encoding_spread` 用于将预测的概率分布转换为 one-hot 编码，并将最大值的前后两个值也置为 1，以增强模型的稳定性。

      函数 `get_pred_top2_direction` 获取预测概率分布中的前两个最大值的索引，然后将这两个索引值减去1，并组成一个张量返回，这个张量表示预测的角度范围。

      函数 `calc_angle_diff` 计算预测角度和真实角度(注意传入的预测角度和真实角度都是数组)之间的差异。它首先根据预测(取top2)获取预测的角度范围，然后根据真实概率(取top2)获取真实的角度范围。最后计算平均差异。


  - **heads**

    - **bev_encoder.py**

      这个文件较为**重要**。

      首先定义了 `class Up(nn.Module)`，应当是上采样，是一个小模块。

      然后就是 `class BevEncode(nn.Module)` 的代码了，用于将 BEV 特征进行编码。

      在 `__init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37, return_feature=False)` 方法中，定义了 BEV 编码器的网络结构。通过上采样层将输入的 BEV 特征进行上采样，然后经过一系列的卷积层和批归一化层，最终得到编码后的 BEV 特征、实例嵌入特征和方向预测特征。根据参数 `instance_seg` 和 `direction_pred` 决定是否生成相应的特征。

      在 `forward` 方法中，接收一个 BEV 特征列表 `bev_feature`，然后将其传入网络中进行前向传播。根据参数决定是否生成实例分割、嵌入特征和方向预测。最终返回一个字典，包含编码后的 BEV 特征以及可能的额外信息。

    - **__init__.py**

      ```python
      from .bev_encoder import BevEncode

      __all__ = [
          'BevEncode',
      ]
      ```

  

  - **losses**

    - **hdmapnet_loss.py**

      这里定义3个loss函数。

      这些损失函数还没太研究，等我再细看看。

      **SimpleLoss**：

      此损失函数采用二元交叉熵损失（BCEWithLogitsLoss）。

      初始化参数包括 `pos_weight` 和 `loss_weight`。`pos_weight` 是正样本的权重，`loss_weight` 是损失函数的权重。

      在前向传播过程中，将预测值和目标值传入二元交叉熵损失函数，并乘以 `loss_weight` 返回损失值。

      **DiscriminativeLoss**：

      此损失函数是差分损失函数（Discriminative Loss），用于实例分割任务。

      初始化参数包括 `embed_dim`（嵌入维度）、`delta_v`（变异损失的阈值）、`delta_d`（距离损失的阈值）和 `loss_weight_cfg`（损失权重配置）。

      在前向传播过程中，输入嵌入特征 `embedding` 和分割目标 `seg_gt`，计算变异损失、距离损失和正则化损失，然后返回这三种损失的加权和。

      **DirectionLoss**：

      此损失函数是用于方向预测任务的损失函数。

      初始化参数包括 `loss_weight`，即损失函数的权重。

      在前向传播过程中，输入预测的方向值 `direction` 和方向的目标值 `direction_mask`，计算二元交叉熵损失（BCELoss）。

      损失值乘以车道掩码 `lane_mask` 后取平均，再乘以 loss_weight 返回。

    - **__init__.py**

      ```python
      from .hdmapnet_loss import SimpleLoss, DiscriminativeLoss, DirectionLoss

      __all__ = [
          'SimpleLoss', 'DiscriminativeLoss', 'DirectionLoss'
      ]
      ```

  - **mapers**

      - **__init__.py**

      ```python
      from .original_hdmapnet import NeuralMapPrior
      from .set_epoch_info_hook import SetEpochInfoHook

      __all__ = [
          'NeuralMapPrior',
      ]
      ```

    - **base_mapper.py**

      定义了 `class BaseMapper(nn.Module, metaclass=ABCMeta)`，是一个基类。

    - **loss_utils.py**

      定义了 `class HdmapNetLosses(torch.nn.Module)`，用于处理损失函数。

      `__init__` 方法， `build_loss` 函数构建了三个损失函数：

      - `self.dir_loss`：方向预测损失函数。
      - `self.embed_loss`：嵌入特征损失函数。
      - `self.seg_loss`：分割损失函数。
      
      还保存了一些其他配置信息，如角度类别数量 `self.angle_class` 和是否进行方向预测 `self.direction_pred`。

      `forward` 方法：
      
      接受输入：预测结果 `preds` 和真实标签 `gts`。
      
      分别计算了分割损失、嵌入特征损失和方向预测损失（如果进行方向预测），将各项损失加权求和作为最终的总损失。
      
      函数的返回值是总损失值 `loss` 和损失日志信息 `log_vars`。

    - **map_global_memory.py**

      首先，从 `project.neural_map_prior.map_tiles.lane_render` 引入了大量的辅助函数。

      然后就是定义一个巨大的类：`class MapGlobalMemory(object)`

      `__init__`方法接受两个参数 `map_attribute` 和 `bev_attribute`，分别表示地图属性和 BEV 属性。
    
      首先保存了地图属性和 BEV 属性；然后根据配置信息，生成训练和验证数据的信息列表和城市列表；根据城市划分的方式或新数据划分的方式，确定了训练和验证数据的列表，并保存了一些数据统计信息；接着根据配置信息和数据信息生成了一些辅助信息，如 `train_gpu2city` 和 `val_gpu2city`。最后，初始化了一些空字典和参数，用于后续的数据处理和计算。

      空字典：

      ```python
      self.map_slice_float_dict = {}
      self.map_slice_int_dict = {}
      self.map_center_dis_dict = {}
      self.map_slice_onehot_dict = {}
      ```

      之后就是挂在这个类之下的函数：

      `gen_token2map_info(self, split)`: 根据给定的数据集划分（如训练集或验证集），加载对应的数据信息和城市列表。如果批量大小不为1，则确保数据信息和城市列表长度一致，并且保证可以被批量大小整除后返回。

      `gen_map_slice_int(map_attribute)`: 根据 `map_attribute` 生成一个新的 `map_attribute`，其中数据类型设置为int16，嵌入维度设置为1。

      `reset_define_map(self, epoch, gpu_id, dataset, map_slices_name, map_attribute_func=None)`: 根据给定的地图片段名称，重置地图，并根据地图属性函数对地图属性进行更新。

      `check_epoch(self, epoch, dataset)`: 检查当前训练/验证的周期是否与之前记录的周期不同，如果不同则更新并返回True。

      `reset_map(self, epoch, gpu_id, dataset, map_slices_dict, map_attribute)`: 重置地图，并根据给定的地图切片字典、地图属性以及训练/验证的数据集、GPU ID等信息生成新的地图。

      `gen_map_info(self, token, split)`: 根据给定的数据集划分和token生成地图信息，包括城市名称、地图索引、地图切片最小边界和最大边界。

      `take_map_prior(self, bev_feature, token, img_meta, dataset, trans)`: 根据给定的token获取地图优先信息，并根据变换将BEV特征投影到地图上。

      `replace_map_prior(self, bev_feature, token, img_metas, dataset, trans)`: 根据给定的token替换地图优先信息，并根据变换将BEV特征投影到地图上。

    - **original_hdmapnet_baseline.py**

      **主干部分**，定义了 `class NeuralMapPrior(BaseMapper)`：

      - `__init__` 方法接受一系列参数来初始化模型，包括地图 BEV 属性 (map_bev_attrs)、距离配置 (dist_cfg)、图像的骨干网络 (img_backbone)、图像的颈部网络 (img_neck)、视图变换配置 (view_transformation_cfg)、头部网络配置 (head_cfg)、损失配置 (loss_cfg)、地图属性 (map_attribute)、是否开启 NMP (open_nmp) 以及其他参数。

      - `set_epoch` 方法用于设置当前的训练周期。

      - `extract_img_feat` 方法用于提取图像特征。它接受图像作为输入，并返回**不同尺度的特征图**。（**重要**）

      - `flatte_feat_and_transpose` 方法用于将特征张量展平并转置，以便进行后续处理。

      - `reshape2bev` 方法用于将特征张量重塑为 BEV 格式。

      - `get_bev_pos` 方法用于获取 BEV 的位置编码。

      - `forward_single` 方法用于进行**单个样本的前向传播**。首先从输入图像中提取图像特征，然后通过视图变换模块将其转换为 BEV 特征。如果开启了神经地图先验 (NMP)，则从全局内存中获取**先验 BEV**。接下来，对当前 BEV 和先验 BEV 进行处理，并通过**卷积 GRU 网络**(定义在`modules/gru_fusion.py`)进行更新。最后，将更新后的 BEV 特征送入头部网络进行预测。

      - `reset_map` 方法用于重置地图。

      - `forward_train` 方法用于在训练过程中进行前向传播。首先从输入中获取图像数据和栅格化的 Ground Truth。如果开启了神经地图先验 (NMP)，则在训练阶段重置地图。然后调用 `forward_single` 方法进行单个样本的前向传播，并计算损失。最后返回损失值、日志变量和批量大小。

      - `forward_test` 方法用于在测试过程中进行前向传播。与训练过程类似，首先获取图像数据和图像元数据。如果开启了神经地图先验 (NMP)，则在测试阶段同样重置地图。然后调用 `forward_single` 方法进行单个样本的前向传播，并进行后处理以生成最终的预测结果。

    - **original_hdmapnet_nmp_final.py**

      定义了 `class OriginalHDMapNet(BaseMapper)`：

      仅有`forward_single`函数和上一个文件有变化：内容增加了。

      在`baseline`版本中，先提取图像特征 `imgs_feats`，然后根据是否开启神经地图先验 `open_nmp` 来处理 BEV 特征。如果开启了神经地图先验，会定义 `cur_bev` 和 `prior_bev`，然后通过全局内存获取 `prior_bev`，并使用卷积 GRU 网络对 `prior_bev` 和 `cur_bev` 进行更新。更新后的结果被展平为 `bev_feats`，然后传递给头部网络进行预测。

      在`final`版本中不再区分 `cur_bev` 和 `prior_bev`，只定义了 `prior_bev`。然后同样通过全局内存获取 `prior_bev`。接下来，处理 BEV 特征，如果开启了神经地图先验，则对 cur_bev 和 `prior_bev` 进行进一步处理，然后通过卷积 GRU 网络进行更新。更新后的结果也被展平为 bev_feats，然后传递给头部网络进行预测。

    - **original_hdmapnet.py**

      这里

      这个类和final版本里的基本相同，但是增加了`positional_encoding_cur`和`positional_encoding_prior`两个参数。

      多了一个 `get_bev_pos` 方法通过调用 `positional_encoding_cur` 和 `positional_encoding_prior` 来获取当前 BEV 特征和先前 BEV 特征的位置编码，并将其用于后续的神经地图先验更新过程中。

    - **set_epoch_info_hook.py**

      定义了 `class SetEpochInfoHook(Hook)`:

      用于在训练过程中设置模型的epoch信息。这里有大量print debug信息没有删除，看起来像惨烈的古战场。

      `before_train_epoch(runner)`: 在每个训练epoch开始之前执行的操作。从模型中移除了`map_slice_float_dict`和`map_slice_int_dict`；

      `after_train_epoch(runner)`: 在每个训练epoch结束之后执行的操作。同样地，从模型中移除了`map_slice_float_dict`和`map_slice_int_dict`；

      此外，还定义了一个辅助方法 `_should_evaluate`(runner)，用于判断是否应该执行评估。

      Here is the rule to judge whether to perform evaluation:
        
      1. It will not perform evaluation during the epoch/iteration interval, which is determined by ``self.interval``.

      2. It will not perform evaluation if the start time is larger than current time.

      3. It will not perform evaluation when current time is larger than the start time but during epoch/iteration interval.

      Returns:

      bool: The flag indicating whether to perform evaluation.



- **modules**

    - **custom_base_transformer_layer.py**

    - **decoder.py**

    - **deformable_transformer.py**

    - **encoder.py**

    - **gru_fusion.py**

    - **__init__.py**

    - **multi_scale_deformable_attn_function.py**

    - **prior_cross_attention.py**

    - **spatial_cross_attention.py**

    - **temporal_self_attention.py**

      - **transformer.py**

      - **utils.py**

      - **window_cross_attention.py**

- **view_transformation**

  - **bevformer.py**

  - **hdmapnet.py**

  - **homography.py**

  - **__init__.py**

  - **lss.py**
