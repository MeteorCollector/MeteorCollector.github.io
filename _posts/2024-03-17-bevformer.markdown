---
layout: post
title:  "BEVFormer代码阅读"
date:   2024-03-17 00:23:00 +0800
categories: posts
tag: cv
---

# BEVFormer

本篇笔记是代码精读，并不涉及理论分析。

代码仓库：[github](https://github.com/fundamentalvision/BEVFormer)

仍在施工中。

## Tree

```
.
├── docs
│   ├── can_bus.ipynb
│   ├── getting_started.md
│   ├── install.md
│   └── prepare_dataset.md
├── figs
│   ├── arch.png
│   └── sota_results.png
├── LICENSE
├── projects
│   ├── configs
│   │   ├── _base_
│   │   │   ├── datasets
│   │   │   │   ├── coco_instance.py
│   │   │   │   ├── kitti-3d-3class.py
│   │   │   │   ├── kitti-3d-car.py
│   │   │   │   ├── lyft-3d.py
│   │   │   │   ├── nuim_instance.py
│   │   │   │   ├── nus-3d.py
│   │   │   │   ├── nus-mono3d.py
│   │   │   │   ├── range100_lyft-3d.py
│   │   │   │   ├── s3dis-3d-5class.py
│   │   │   │   ├── s3dis_seg-3d-13class.py
│   │   │   │   ├── scannet-3d-18class.py
│   │   │   │   ├── scannet_seg-3d-20class.py
│   │   │   │   ├── sunrgbd-3d-10class.py
│   │   │   │   ├── waymoD5-3d-3class.py
│   │   │   │   └── waymoD5-3d-car.py
│   │   │   ├── default_runtime.py
│   │   │   ├── models
│   │   │   │   ├── 3dssd.py
│   │   │   │   ├── cascade_mask_rcnn_r50_fpn.py
│   │   │   │   ├── centerpoint_01voxel_second_secfpn_nus.py
│   │   │   │   ├── centerpoint_02pillar_second_secfpn_nus.py
│   │   │   │   ├── fcos3d.py
│   │   │   │   ├── groupfree3d.py
│   │   │   │   ├── h3dnet.py
│   │   │   │   ├── hv_pointpillars_fpn_lyft.py
│   │   │   │   ├── hv_pointpillars_fpn_nus.py
│   │   │   │   ├── hv_pointpillars_fpn_range100_lyft.py
│   │   │   │   ├── hv_pointpillars_secfpn_kitti.py
│   │   │   │   ├── hv_pointpillars_secfpn_waymo.py
│   │   │   │   ├── hv_second_secfpn_kitti.py
│   │   │   │   ├── hv_second_secfpn_waymo.py
│   │   │   │   ├── imvotenet_image.py
│   │   │   │   ├── mask_rcnn_r50_fpn.py
│   │   │   │   ├── paconv_cuda_ssg.py
│   │   │   │   ├── paconv_ssg.py
│   │   │   │   ├── parta2.py
│   │   │   │   ├── pointnet2_msg.py
│   │   │   │   ├── pointnet2_ssg.py
│   │   │   │   └── votenet.py
│   │   │   └── schedules
│   │   │       ├── cosine.py
│   │   │       ├── cyclic_20e.py
│   │   │       ├── cyclic_40e.py
│   │   │       ├── mmdet_schedule_1x.py
│   │   │       ├── schedule_2x.py
│   │   │       ├── schedule_3x.py
│   │   │       ├── seg_cosine_150e.py
│   │   │       ├── seg_cosine_200e.py
│   │   │       └── seg_cosine_50e.py
│   │   ├── bevformer
│   │   │   ├── bevformer_base.py
│   │   │   ├── bevformer_small.py
│   │   │   └── bevformer_tiny.py
│   │   ├── bevformer_fp16
│   │   │   └── bevformer_tiny_fp16.py
│   │   ├── bevformerv2
│   │   │   ├── bevformerv2-r50-t1-24ep.py
│   │   │   ├── bevformerv2-r50-t1-48ep.py
│   │   │   ├── bevformerv2-r50-t1-base-24ep.py
│   │   │   ├── bevformerv2-r50-t1-base-48ep.py
│   │   │   ├── bevformerv2-r50-t2-24ep.py
│   │   │   ├── bevformerv2-r50-t2-48ep.py
│   │   │   └── bevformerv2-r50-t8-24ep.py
│   │   └── datasets
│   │       ├── custom_lyft-3d.py
│   │       ├── custom_nus-3d.py
│   │       └── custom_waymo-3d.py
│   ├── __init__.py
│   └── mmdet3d_plugin
│       ├── bevformer
│       │   ├── apis
│       │   │   ├── __init__.py
│       │   │   ├── mmdet_train.py
│       │   │   ├── test.py
│       │   │   └── train.py
│       │   ├── dense_heads
│       │   │   ├── bevformer_head.py
│       │   │   ├── bev_head.py
│       │   │   └── __init__.py
│       │   ├── detectors
│       │   │   ├── bevformer_fp16.py
│       │   │   ├── bevformer.py
│       │   │   ├── bevformerV2.py
│       │   │   └── __init__.py
│       │   ├── hooks
│       │   │   ├── custom_hooks.py
│       │   │   └── __init__.py
│       │   ├── __init__.py
│       │   ├── modules
│       │   │   ├── custom_base_transformer_layer.py
│       │   │   ├── decoder.py
│       │   │   ├── encoder.py
│       │   │   ├── group_attention.py
│       │   │   ├── __init__.py
│       │   │   ├── multi_scale_deformable_attn_function.py
│       │   │   ├── spatial_cross_attention.py
│       │   │   ├── temporal_self_attention.py
│       │   │   ├── transformer.py
│       │   │   └── transformerV2.py
│       │   └── runner
│       │       ├── epoch_based_runner.py
│       │       └── __init__.py
│       ├── core
│       │   ├── bbox
│       │   │   ├── assigners
│       │   │   │   ├── hungarian_assigner_3d.py
│       │   │   │   └── __init__.py
│       │   │   ├── coders
│       │   │   │   ├── __init__.py
│       │   │   │   └── nms_free_coder.py
│       │   │   ├── match_costs
│       │   │   │   ├── __init__.py
│       │   │   │   └── match_cost.py
│       │   │   └── util.py
│       │   └── evaluation
│       │       ├── eval_hooks.py
│       │       ├── __init__.py
│       │       └── kitti2waymo.py
│       ├── datasets
│       │   ├── builder.py
│       │   ├── __init__.py
│       │   ├── nuscenes_dataset.py
│       │   ├── nuscenes_dataset_v2.py
│       │   ├── nuscenes_mono_dataset.py
│       │   ├── nuscnes_eval.py
│       │   ├── pipelines
│       │   │   ├── augmentation.py
│       │   │   ├── dd3d_mapper.py
│       │   │   ├── formating.py
│       │   │   ├── __init__.py
│       │   │   ├── loading.py
│       │   │   └── transform_3d.py
│       │   └── samplers
│       │       ├── distributed_sampler.py
│       │       ├── group_sampler.py
│       │       ├── __init__.py
│       │       └── sampler.py
│       ├── dd3d
│       │   ├── datasets
│       │   │   ├── __init__.py
│       │   │   ├── nuscenes.py
│       │   │   └── transform_utils.py
│       │   ├── __init__.py
│       │   ├── layers
│       │   │   ├── iou_loss.py
│       │   │   ├── normalization.py
│       │   │   └── smooth_l1_loss.py
│       │   ├── modeling
│       │   │   ├── core.py
│       │   │   ├── disentangled_box3d_loss.py
│       │   │   ├── fcos2d.py
│       │   │   ├── fcos3d.py
│       │   │   ├── __init__.py
│       │   │   ├── nuscenes_dd3d.py
│       │   │   └── prepare_targets.py
│       │   ├── structures
│       │   │   ├── boxes3d.py
│       │   │   ├── image_list.py
│       │   │   ├── __init__.py
│       │   │   ├── pose.py
│       │   │   └── transform3d.py
│       │   └── utils
│       │       ├── comm.py
│       │       ├── geometry.py
│       │       ├── tasks.py
│       │       ├── tensor2d.py
│       │       └── visualization.py
│       ├── __init__.py
│       └── models
│           ├── backbones
│           │   ├── __init__.py
│           │   └── vovnet.py
│           ├── hooks
│           │   ├── hooks.py
│           │   └── __init__.py
│           ├── opt
│           │   ├── adamw.py
│           │   └── __init__.py
│           └── utils
│               ├── bricks.py
│               ├── grid_mask.py
│               ├── __init__.py
│               ├── position_embedding.py
│               └── visual.py
├── README.md
├── tools
│   ├── analysis_tools
│   │   ├── analyze_logs.py
│   │   ├── benchmark.py
│   │   ├── get_params.py
│   │   ├── __init__.py
│   │   └── visual.py
│   ├── create_data.py
│   ├── data_converter
│   │   ├── create_gt_database.py
│   │   ├── indoor_converter.py
│   │   ├── __init__.py
│   │   ├── kitti_converter.py
│   │   ├── kitti_data_utils.py
│   │   ├── lyft_converter.py
│   │   ├── lyft_data_fixer.py
│   │   ├── nuimage_converter.py
│   │   ├── nuscenes_converter.py
│   │   ├── s3dis_data_utils.py
│   │   ├── scannet_data_utils.py
│   │   ├── sunrgbd_data_utils.py
│   │   └── waymo_converter.py
│   ├── dist_test.sh
│   ├── dist_train.sh
│   ├── fp16
│   │   ├── dist_train.sh
│   │   └── train.py
│   ├── misc
│   │   ├── browse_dataset.py
│   │   ├── fuse_conv_bn.py
│   │   ├── print_config.py
│   │   └── visualize_results.py
│   ├── model_converters
│   │   ├── convert_votenet_checkpoints.py
│   │   ├── publish_model.py
│   │   └── regnet2mmdet.py
│   ├── test.py
│   └── train.py
└── tree.txt

46 directories, 191 files
```
