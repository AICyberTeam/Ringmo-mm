<div align="center">
  <img src="resources/ringmo.jpg" width="800"/>
</div>



## 介绍

RingMoMultimodal 是是空天院AI赛博试验室开发的应用于遥感领域的多模态工具集。

主分支代码目前支持 **PyTorch 1.6 以上**的版本。

<details open>
<summary><b>主要特性</b></summary>

- **灵眸大模型支持**

  RingMoMultimodal 支持空天灵眸大模型，并实现灵眸大模型权重分层次调用。

- **配置化设计**

  RingMoMultimodal 将多模态任务不同的模块组件，通过组合不同的模块组件。通过配置文件，可以实现多

- **多种类多模态任务支持**

  RingMoMultimodal 实现视觉定位跨模态任务。并涵盖了会将AI赛博试验室最新的多模态研究结果。

</details>

## 安装

RingMoMultimodal 依赖 [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) 、 [MMDetection](https://github.com/open-mmlab/mmdetection)，以下是安装的简要步骤。
更详细的安装指南请参考 [安装文档](https://mmrotate.readthedocs.io/zh_CN/latest/install.html)。

```shell
conda create -n ringmo-multimodal python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate ringmo-multimodal
pip install openmim
mim install mmcv-full
mim install mmdet
min install mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

## 模型手册

### 训练命令
```shell
cd ringmoMultiModal
export PYTHONPATH=$(pwd):$PYTHONPATH
python tools/train.py configs/faoa_darknet_bert.py 
```
### 测试命令
```shell
cd ringmoMultiModal
export PYTHONPATH=$(pwd):$PYTHONPATH
python tools/test.py configs/faoa_darknet_bert.py –work_dir ./work_dirs/visual_grounding/latest.pth –out ./pickle.pkl
```

### 可视化命令
```shell
cd ringmoMultiModal
export PYTHONPATH=$(pwd):$PYTHONPATH
python tools/test.py configs/faoa_darknet_bert.py –work_dir ./work_dirs/visual_grounding/latest.pth –show_dir ./demo/images/
```
## 高级配置

### 分层调用示例

- 多模态主干网络（MultimodalBackbone）配置 
  多模态主干网络简易如下：
  ```python
   multimodal_backbone=dict(
        type="SimpleMultiModal",
        backbone_left={
            **ringmo1b['backbone'],
             "out_indices": [3]},
        backbone_right=dict(
            type="Bert",
            init_cfg=dict(
                bert_model="bert-base-uncased.tar.gz",
                tuned=False
            )
        ),
        top_bridge=dict(
            type="VLFBridge",
        ),
        bridges=[dict(type="xxx"), dict(type="xxx")]
  )
  ```
## Citation
If you feel this code helpful or use this code or dataset, please cite it as
```
C.Li  et al., Injecting Linguistic Into Visual Backbone: Query-Aware Multimodal Fusion Network for Remote Sensing Visual Grounding, [J]. IEEE Transactions on Geoscience and Remote Sensing, 2024.
```

