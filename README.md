# Enhancing scene graph generation via Hybrid Co-Attention and Predicate Reweighting for long-tail robustness (ReBalance-HCA)

This is the PyTorch implementation for **ReBalance-HCA** proposed in the paper **Enhancing scene graph generation via Hybrid Co-Attention and Predicate Reweighting for long-tail robustness**. The scene graph generation task aims to generate a set of triplet <subject, predicate, object> and construct a graph structure for an image.

## Architecture

![framework](framework.png)

## Environments

* **python 3.8**
* **pytorch 2.4.0**

## DATASET
The following is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs).

Note that our codebase intends to support attribute-head too, so our ```VG-SGG.h5``` and ```VG-SGG-dicts.json``` are different with their original versions in [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). We add attribute information and rename them to be ```VG-SGG-with-attri.h5``` and ```VG-SGG-dicts-with-attri.json```. The code we use to generate them is located at ```datasets/vg/generate_attribute_labels.py```. Consistent with the conventional approach described in 'Unbiased Scene Graph Generation from Biased Training', we disable the attribute head during detector pretraining and relationship prediction to ensure fair comparison, mirroring this codebase's default configuration.

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Download the [scene graphs](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779871&authkey=AA33n7BRpB1xa3I) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `maskrcnn_benchmark/config/paths_catelog.py`.

## Setup 
Follow the [instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to install and use the code. Follow some [scripts](https://github.com/ZhuGeKongKong/SSG-G2S/tree/main/scripts) for training and testing.
The training process is divided into two stages: **Training the common SGG model** and **Finetuning the informative SGG model**.

## 
