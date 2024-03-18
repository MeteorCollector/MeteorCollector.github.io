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

Getting started: evaluation test 第一个需要运行的脚本。

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

暂略

### __init__.py

### neural_map_prior

#### ckpts

需要用户自己创建一个 `ckpts` 文件夹，在这里存储训练的 `checkpoint` 文件

#### datasets

- **base_dataset.py**

- **evaluation**

  - **chamfer_dist.py**

  - **eval_dataloader.py**

  - **hdmap_eval.py**

  - **__init__.py**

  - **iou.py**

  - **precision_recall**

    - **average_precision_det.py**

    - **tgtg_chamfer.py**

    - **tgtg.py**

  - **rasterize.py**

  - **utils.py**

  - **vectorized_map.py**

- **__init__.py**

- **nuscenes_utils**

  - **base_nusc_dataset.py**

  - **hdmapnet.py**

  - **__init__.py**

  - **map_api.py**

  - **nuscene.py**

  - **utils.py**
  
- **nuscenes_dataset.py**

- **pipelines**

  - **formating.py**

  - **__init__.py**

  - **loading.py**

  - **map_transform.py**

- **data_utils**

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

- **__init__.py**

- **map_tiles**

  - **__init__.py**

  - **lane_render.py**

  - **local_multi_trips.py**

  - **nusc_split.py**

- **models**

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
