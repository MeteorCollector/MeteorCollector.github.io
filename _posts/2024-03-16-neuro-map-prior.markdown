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

    `gen_vectorized_samples(...)`: 用于生成矢量化的局部地图数据。它接受位置信息、车辆到全局坐标系的平移和旋转信息，并返回过滤后的矢量数据。在该方法中，通过调用 `get_map_geom` 方法获取指定要素的几何信息，然后通过一系列方法将几何信息转换为矢量数据。最终将所有有效的矢量数据存储在 `filtered_vectors` 中并返回。

    `get_map_geom(...)`: 这个方法用于获取指定范围内、指定 `layer` 的几何信息。它接受 `patch` 的尺寸和角度、`layer` 的类别和问询的位置，并返回符合条件目标的几何信息。具体是通过调用 `NuScenesMapExplorer` 来实现的。

    `_one_type_line_geom_to_vectors(...)`: 这是一个辅助方法，用于将特定类型的线条几何转换为矢量。它接受线条几何信息，并返回矢量化的线条数据。

    `poly_geoms_to_vectors(...)`, `line_geoms_to_vectors(...)`, `ped_geoms_to_vectors(...)`: 这三个方法分别用于将多边形、线条和人行横道几何转换为矢量数据。它们通过调用 `_one_type_line_geom_to_vectors` 方法实现。

    `get_ped_crossing_line(...)`: 这个方法用于获取人行横道的线条信息。它通过解析地图中的人行横道多边形信息来获取相应的线条，并返回这些线条的列表。

    `sample_pts_from_line(...)`: 这个方法用于从线条中抽样点。它接受线条几何信息，并返回采样后的点坐标以及有效点的数量。


- **nuscenes_utils**

  - **base_nusc_dataset.py**

  - **hdmapnet.py**

  - **__init__.py**

  - **map_api.py**

  - **nuscene.py**

  - **utils.py**

- **pipelines**

  - **formating.py**

  - **__init__.py**

  - **loading.py**

  - **map_transform.py**

#### data_utils

  - **boston_split_gen**

    - **boston_data_split.py**

    - **detect_trip_overlap.py**

    - **doc.md**

    - **order_overlapping_trips_by_timestamp.py**

    - **rotate_iou.py**

    - **trip_overlap_val_h60_w30_thr0**

      - **samples_poses_array.pkl**

      - **samples_tokens.pkl**

      - **samples_poses_array.pkl**

      - **samples_trans_array.pkl**

      - **split_scenes.pkl**

      - **trip_overlap_val_60_30_1.pkl**

  - **nusc_city_infos.py**

#### map_tiles

  - **__init__.py**

  - **lane_render.py**

  - **local_multi_trips.py**

  - **nusc_split.py**

#### models

  - **hdmapnet_utils**

    - **angle_diff.py**

    - **__init.py__**

  - **heads**

    - **bev_encoder.py**

    - **__init__.py**

  - **__init.py__**

  - **losses**

    - **hdmapnet_loss.py**

    - **__init__.py**

  - **mapers**

    - **base_mapper.py**

    - **__init__.py**

    - **loss_utils.py**

    - **map_global_memory.py**

    - **original_hdmapnet_baseline.py**

    - **original_hdmapnet_nmp_final.py**

    - **original_hdmapnet.py**

    - **set_epoch_info_hook.py**

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
