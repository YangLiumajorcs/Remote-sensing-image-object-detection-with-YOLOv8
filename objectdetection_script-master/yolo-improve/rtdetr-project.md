# 目前自带的一些改进方案(持续更新)

# 为了感谢各位对RTDETR项目的支持,本项目的赠品是yolov5-PAGCP通道剪枝算法.[具体使用教程](https://www.bilibili.com/video/BV1yh4y1Z7vz/)

# 自带的一些文件说明
1. train.py
    训练模型的脚本
2. main_profile.py
    输出模型和模型每一层的参数,计算量的脚本(rtdetr-l和rtdetr-x因为thop库的问题,没办法正常输出每一层的参数和计算量和时间)
3. val.py
    使用训练好的模型计算指标的脚本
4. detect.py
    推理的脚本
5. track.py
    跟踪推理的脚本
6. heatmap.py
    生成热力图的脚本
7. get_FPS.py
    计算模型储存大小、模型推理时间、FPS的脚本
8. get_COCO_metrice.py
    计算COCO指标的脚本
9. plot_result.py
    绘制曲线对比图的脚本
10. get_model_erf.py
    绘制模型的有效感受野.[视频链接](https://www.bilibili.com/video/BV1Gx4y1v7ZZ/)

# RT-DETR基准模型

1. ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml(有预训练权重COCO+Objects365,来自RTDETR-Pytorch版本的移植)

    rtdetr-r18 summary: 421 layers, 20184464 parameters, 20184464 gradients, 58.6 GFLOPs
2. ultralytics/cfg/models/rt-detr/rtdetr-r34.yaml(有预训练权重COCO,来自RTDETR-Pytorch版本的移植)

    rtdetr-r34 summary: 525 layers, 31441668 parameters, 31441668 gradients, 90.6 GFLOPs
3. ultralytics/cfg/models/rt-detr/rtdetr-r50-m.yaml(有预训练权重COCO,来自RTDETR-Pytorch版本的移植)

    rtdetr-r50-m summary: 637 layers, 36647020 parameters, 36647020 gradients, 98.3 GFLOPs
4. ultralytics/cfg/models/rt-detr/rtdetr-r50.yaml(有预训练权重COCO+Objects365,来自RTDETR-Pytorch版本的移植)

    rtdetr-r50 summary: 629 layers, 42944620 parameters, 42944620 gradients, 134.8 GFLOPs
5. ultralytics/cfg/models/rt-detr/rtdetr-r101.yaml

    rtdetr-r101 summary: 867 layers, 76661740 parameters, 76661740 gradients, 257.7 GFLOPs
6. ultralytics/cfg/models/rt-detr/rtdetr-l.yaml(有预训练权重)

    rtdetr-l summary: 673 layers, 32970732 parameters, 32970732 gradients, 108.3 GFLOPs
7. ultralytics/cfg/models/rt-detr/rtdetr-x.yaml(有预训练权重)

    rtdetr-x summary: 867 layers, 67468108 parameters, 67468108 gradients, 232.7 GFLOPs
# 专栏改进汇总

### 二次创新系列
1. ultralytics/cfg/models/rt-detr/rtdetr-DCNV2-Dynamic.yaml

    使用自研可变形卷积DCNV2-Dynamic改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-MPCA与DCNV2_Dynamic的说明)
2. ultralytics/cfg/models/rt-detr/rtdetr-iRMB-Cascaded.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-20231119更新说明)
3. ultralytics/cfg/models/rt-detr/rtdetr-PConv-Rep.yaml

    使用[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv进行二次创新后改进resnet18-backbone中的BasicBlock.
4. ultralytics/cfg/models/rt-detr/rtdetr-Faster-Rep.yaml

    使用[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新后改进resnet18-backbone中的BasicBlock.
5. ultralytics/cfg/models/rt-detr/rtdetr-Faster-EMA.yaml

    使用[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新后改进resnet18-backbone中的BasicBlock.
6. ultralytics/cfg/models/rt-detr/rtdetr-Faster-Rep-EMA.yaml
    
    使用[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv和[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)对[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新后改进resnet18-backbone中的BasicBlock.
7. ultralytics/cfg/models/rt-detr/rtdetr-DWRC3-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)进行二次创新改进rtdetr.
8. ultralytics/cfg/models/rt-detr/rtdetr-ASF-P2.yaml

    在ultralytics/cfg/models/rt-detr/rtdetr-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.
9. ultralytics/cfg/models/rt-detr/rtdetr-slimneck-ASF.yaml

    使用[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中的VoVGSCSP\VoVGSCSPC和GSConv和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进rtdetr中的CCFM.
10. ultralytics/cfg/models/rt-detr/rtdetr-goldyolo-asf.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行改进特征融合模块.
11. ultralytics/cfg/models/rt-detr/rtdetr-HSPAN.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进RTDETR中的CCFM.
12. ultralytics/cfg/models/rt-detr/rtdetr-ASF-Dynamic.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion的上采样模块得到Dynamic Sample Attentional Scale Sequence Fusion改进CCFM.
13. ultralytics/cfg/models/rt-detr/rtdetr-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进resnet18-backbone中的BasicBlock.
14. ultralytics/cfg/models/rt-detr/rtdetr-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进resnet18-backbone中的BasicBlock.
15. ultralytics/cfg/models/rt-detr/rtdetr-DBBNCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[Diverse Branch Block CVPR2021](https://arxiv.org/abs/2103.13425)进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

16. ultralytics/cfg/models/rt-detr/rtdetr-OREPANCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

17. ultralytics/cfg/models/rt-detr/rtdetr-DRBNCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

18. ultralytics/cfg/models/rt-detr/rtdetr-Conv3XCNCSPELAN.yaml

    在rtdetr-RepNCSPELAN.yaml使用[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC进行二次创新.(详细介绍请看百度云视频-20240225更新说明)

19. ultralytics/cfg/models/rt-detr/rtdetr-ELA-HSFPN.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN.

20. ultralytics/cfg/models/rt-detr/rtdetr-CA-HSFPN.yaml

    使用[Coordinate Attention CVPR2021](https://github.com/houqb/CoordAttention)改进HSFPN.

21. ultralytics/cfg/models/rt-detr/rtdetr-RepNCSPELAN-CAA.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块改进RepNCSPELAN.

22. ultralytics/cfg/models/rt-detr/rtdetr-CAA-HSFPN.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块HSFPN.

### 自研系列
1. ultralytics/cfg/models/rt-detr/rtdetr-PACAPN.yaml

    自研结构, Parallel Atrous Convolution Attention Pyramid Network, PAC-APN
    1. 并行(上/下)采样分支可为网络提供多条特征提取途径，丰富特征表达的多样性、再结合gate机制对采样后的特征进行特征选择，强化更有意义的特征，抑制冗余或不相关的特征，提升特征表达的有效性。
    2. PAC模块通过使用具有不同膨胀率的并行空洞卷积，能够有效地提取不同尺度的特征。这使得网络能够捕捉数据中局部和上下文信息，提高其表示复杂模式的能力。

2. ultralytics/cfg/models/rt-detr/rtdetr-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.

3. ultralytics/cfg/models/rt-detr/rtdetr-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.

### BackBone系列
1. ultralytics/cfg/models/rt-detr/rt-detr-timm.yaml

    使用[timm](https://github.com/huggingface/pytorch-image-models)库系列的主干替换rtdetr的backbone.(基本支持现有CNN模型)
2. ultralytics/cfg/models/rt-detr/rt-detr-fasternet.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)替换rtdetr的backbone.
3. ultralytics/cfg/models/rt-detr/rt-detr-EfficientViT.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)替换rtdetr的backbone.
4. ultralytics/cfg/models/rt-detr/rtdetr-convnextv2.yaml

    使用[ConvNextV2 2023](https://github.com/facebookresearch/ConvNeXt-V2)替换rtdetr的backbone.
5. ultralytics/cfg/models/rt-detr/rtdetr-EfficientFormerv2.yaml

    使用[EfficientFormerv2 2022](https://github.com/snap-research/EfficientFormer)替换rtdetr的backbone.
6. ultralytics/cfg/models/rt-detr/rtdetr-repvit.yaml

    使用[RepViT ICCV2023](https://github.com/THU-MIG/RepViT)替换rtdetr的backbone.
7. ultralytics/cfg/models/rt-detr/rtdetr-CSwomTramsformer.yaml

    使用[CSwinTramsformer CVPR2022](https://github.com/microsoft/CSWin-Transformer)替换rtdetr的backbone.
8. ultralytics/cfg/models/rt-detr/rtdetr-VanillaNet.yaml

    使用[VanillaNet 2023](https://github.com/huawei-noah/VanillaNet)替换rtdetr的backbone.
9. ultralytics/cfg/models/rt-detr/rtdetr-SwinTransformer.yaml

    使用[SwinTransformer ICCV2021](https://github.com/microsoft/Swin-Transformer)替换rtdetr的backbone.
10. ultralytics/cfg/models/rt-detr/rtdetr-lsknet.yaml

    使用[LSKNet ICCV2023](https://github.com/zcablii/LSKNet)替换rtdetr的backbone.
11. ultralytics/cfg/models/rt-detr/rt-detr-unireplknet.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)替换rtdetr的backbone.
12. ultralytics/cfg/models/rt-detr/rtdetr-TransNeXt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)改进rtdetr的backbone.
13. ultralytics/cfg/models/rt-detr/rtdetr-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN和ADown进行改进RTDETR-R18.
14. ultralytics/cfg/models/rt-detr/rtdetr-rmt.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)改进rtdetr的主干.
15. ultralytics/cfg/models/rt-detr/rtdetr-C2f-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块和C2f改进backbone.
16. ultralytics/cfg/models/rt-detr/rtdetr-C2f-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C2f.
### AIFI系列
1. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-LPE.yaml

    使用LearnedPositionalEncoding改进AIFI中的位置编码生成.(详细介绍请看百度云视频-20231119更新说明)
2. ultralytics/cfg/models/rt-detr/rtdetr-CascadedGroupAttention.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention改进rtdetr中的AIFI.(详细请看百度云视频-rtdetr-CascadedGroupAttention说明)
3. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-DAttention.yaml

    使用[Vision Transformer with Deformable Attention CVPR2022](https://github.com/LeapLabTHU/DAT)中的DAttention改进AIFI.
4. ultralytics/cfg/models/rt-detr/rtdetr-AIFI-HiLo.yaml

    使用[LITv2](https://github.com/ziplab/LITv2)中具有提取高低频信息的高效注意力对AIFI进行二次改进.

### Neck系列
1. ultralytics/cfg/models/rt-detr/rtdetr-ASF.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion来改进rtdetr.
2. ultralytics/cfg/models/rt-detr/rtdetr-slimneck.yaml

    使用[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中的VoVGSCSP\VoVGSCSPC和GSConv改进rtdetr中的CCFM.
3. ultralytics/cfg/models/rt-detr/rtdetr-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对CCFM中的feature fusion进行改进.
4. ultralytics/cfg/models/rt-detr/rtdetr-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.
5. ultralytics/cfg/models/rt-detr/rtdetr-HSFPN.yaml

    使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进RTDETR中的CCFM.
6. ultralytics/cfg/models/rt-detr/rtdetr-bifpn.yaml

    添加BIFPN到rtdetr-r18中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        block模块选择,具体可看对应百度云视频-20240302更新公告.
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

### Head系列
1. ultralytics/cfg/models/rt-detr/rtdetr-p2.yaml

    添加小目标检测头P2到TransformerDecoderHead中.

### RepC3改进系列
1. ultralytics/cfg/models/rt-detr/rtdetr-DWRC3.yaml

    使用[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块构建DWRC3改进rtdetr.
2. ultralytics/cfg/models/rt-detr/rtdetr-Conv3XCC3.yaml

    使用[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进RepC3.
3. ultralytics/cfg/models/rt-detr/rtdetr-DRBC3.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进RepC3.
4. ultralytics/cfg/models/rt-detr/rtdetr-DBBC3.yaml

    使用[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进RepC3.
5. ultralytics/cfg/models/rt-detr/rtdetr-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进rtdetr-r18.
6. ultralytics/cfg/models/rt-detr/rtdetr-DGCST2.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer与Dynamic Group Convolution Shuffle Module进行结合改进rtdetr-r18.
7. ultralytics/cfg/models/rt-detr/rtdetr-RetBlockC3.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进RepC3.
### ResNet主干中的BasicBlock/BottleNeck改进系列(以下改进BottleNeck基本都有,就不再重复标注)
1. ultralytics/cfg/models/rt-detr/rtdetr-Ortho.yaml

    使用[OrthoNets](https://github.com/hady1011/OrthoNets/tree/main)中的正交通道注意力改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-20231119更新说明)
2. ultralytics/cfg/models/rt-detr/rtdetr-DCNV2.yaml

    使用可变形卷积DCNV2改进resnet18-backbone中的BasicBlock.
3. ultralytics/cfg/models/rt-detr/rtdetr-DCNV3.yaml

    使用可变形卷积[DCNV3 CVPR2023](https://github.com/OpenGVLab/InternImage)改进resnet18-backbone中的BasicBlock.(安装教程请看百度云视频-20231119更新说明)
4. ultralytics/cfg/models/rt-detr/rtdetr-iRMB.yaml

    使用[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB改进resnet18-backbone中的BasicBlock.(详细介绍请看百度云视频-20231119更新说明)
5. ultralytics/cfg/models/rt-detr/rtdetr-DySnake.yaml

    添加[DySnakeConv](https://github.com/YaoleiQi/DSCNet)到resnet18-backbone中的BasicBlock中.
6. ultralytics/cfg/models/rt-detr/rtdetr-PConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv改进resnet18-backbone中的BasicBlock.
7. ultralytics/cfg/models/rt-detr/rtdetr-Faster.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block改进resnet18-backbone中的BasicBlock.
8. ultralytics/cfg/models/rt-detr/rtdetr-AKConv.yaml

    使用[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进resnet18-backbone中的BasicBlock.

9. ultralytics/cfg/models/rt-detr/rtdetr-RFAConv.yaml

    使用[RFAConv 2023](https://github.com/Liuchen1997/RFAConv)改进resnet18-backbone中的BasicBlock.

10. ultralytics/cfg/models/rt-detr/rtdetr-RFCAConv.yaml

    使用[RFCAConv 2023](https://github.com/Liuchen1997/RFAConv)改进resnet18-backbone中的BasicBlock.

11. ultralytics/cfg/models/rt-detr/rtdetr-RFCBAMConv.yaml

    使用[RFCBAMConv 2023](https://github.com/Liuchen1997/RFAConv)改进resnet18-backbone中的BasicBlock.
12. ultralytics/cfg/models/rt-detr/rtdetr-Conv3XC.yaml

    使用[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进resnet18-backbone中的BasicBlock.
13. ultralytics/cfg/models/rt-detr/rtdetr-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进resnet18-backbone中的BasicBlock.
14. ultralytics/cfg/models/rt-detr/rtdetr-DBB.yaml

    使用[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进resnet18-backbone中的BasicBlock.
15. ultralytics/cfg/models/rt-detr/rtdetr-DualConv.yaml

    使用[DualConv](https://github.com/ChipsGuardian/DualConv)改进resnet18-backbone中的BasicBlock.
16. ultralytics/cfg/models/rt-detr/rtdetr-AggregatedAtt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进resnet18中的BasicBlock.(百度云视频-20240106更新说明)
17. ultralytics/cfg/models/rt-detr/rtdetr-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进resnet18中的BasicBlock.
18. ultralytics/cfg/models/rt-detr/rtdetr-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进resnet18中的BasicBlock.
19. ultralytics/cfg/models/rt-detr/rtdetr-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)改进resnet18-backbone中的BasicBlock.
20. ultralytics/cfg/models/rt-detr/rtdetr-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided和Light-weight Context Guided DownSample改进rtdetr-r18.
21. ultralytics/cfg/models/rt-detr/rtdetr-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进resnet18-basicblock.
### 上下采样算子系列
1. ultralytics/cfg/models/rt-detr/rtdetr-DySample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进CCFM中的上采样.
2. ultralytics/cfg/models/rt-detr/rtdetr-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进CCFM中的上采样.
3. ultralytics/cfg/models/rt-detr/rtdetr-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进CCFM的下采样.
4. ultralytics/cfg/models/rt-detr/rtdetr-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample改进rtdetr-r18.

### RT-DETR-L改进系列
1. ultralytics/cfg/models/rt-detr/rtdetr-l-GhostHGNetV2.yaml

    使用GhostConv改进HGNetV2.(详细介绍请看百度云视频-20231109更新说明)

2. ultralytics/cfg/models/rt-detr/rtdetr-l-RepHGNetV2.yaml

    使用RepConv改进HGNetV2.(详细介绍请看百度云视频-20231109更新说明)

3. ultralytics/cfg/models/rt-detr/rtdetr-l-attention.yaml

    添加注意力模块到HGBlock中.(手把手教程请看百度云视频-手把手添加注意力教程)

### 注意力系列
1. EMA
2. SimAM
3. SpatialGroupEnhance
4. BiLevelRoutingAttention, BiLevelRoutingAttention_nchw
5. TripletAttention
6. CoordAtt
7. CBAM
8. BAMBlock
9. EfficientAttention(CloFormer中的注意力)
10. LSKBlock
11. SEAttention
12. CPCA
13. deformable_LKA
14. EffectiveSEModule
15. LSKA
16. SegNext_Attention
17. DAttention(Vision Transformer with Deformable Attention CVPR2022)
18. FocusedLinearAttention(ICCV2023)
19. MLCA
20. TransNeXt_AggregatedAttention
21. HiLo

### IoU系列
1. IoU,GIoU,DIoU,CIoU,EIoU,SIoU(百度云视频-20231125更新说明)
2. MPDIoU[论文链接](https://arxiv.org/pdf/2307.07662.pdf)(百度云视频-20231125更新说明)
3. Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU[论文链接](https://arxiv.org/abs/2311.02877)(百度云视频-20231125更新说明)
4. Inner-MPDIoU(利用Inner-Iou与MPDIou进行二次创新)(百度云视频-20231125更新说明)
5. Normalized Gaussian Wasserstein Distance.[论文链接](https://arxiv.org/abs/2110.13389)(百度云视频-20231125更新说明)
6. Shape-IoU,Inner-Shape-IoU[论文链接](https://arxiv.org/abs/2110.13389)(百度云视频-20240106更新说明)
7. SlideLoss,EMASlideLoss[创新思路](https://www.bilibili.com/video/BV1W14y1i79U/?vd_source=c8452371e7ca510979593165c8d7ac27).[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)(百度云视频-20240113更新说明)
8. Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)(百度云视频-20240113更新说明)
9. Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)(百度云视频-20240113更新说明)
10. Focaler-IoU,Focaler-GIoU,Focaler-DIoU,Focaler-CIoU,Focaler-EIoU,Focaler-SIoU,Focaler-Shape-IoU,Focaler-MPDIoU[论文链接](https://arxiv.org/abs/2401.10525)(百度云视频-20240128更新说明)
11. Focaler-Wise-IoU(v1,v2,v3)(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)[论文链接](https://arxiv.org/abs/2401.10525)(百度云视频-20240128更新说明)
12. Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2,Wise-Powerful-IoU(v1,v2,v3),Wise-Powerful-IoUV2(v1,v2,v3)[论文链接](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640)
13. SlideVarifocalLoss,EMASlideVarifocalLoss[创新思路](https://www.bilibili.com/video/BV1W14y1i79U/?vd_source=c8452371e7ca510979593165c8d7ac27).[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)(百度云视频-20240302更新说明)

### 以Yolov8为基准模型的改进方案
1. ultralytics/cfg/models/yolo-detr/yolov8-detr.yaml

    使用RT-DETR中的TransformerDecoderHead改进yolov8.

2. ultralytics/cfg/models/yolo-detr/yolov8-detr-DWR.yaml

    使用RT-DETR中的TransformerDecoderHead和[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块改进yolov8.

3. ultralytics/cfg/models/yolo-detr/yolov8-detr-fasternet.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)改进yolov8.(支持替换其他主干,请看百度云视频-替换主干示例教程)

4. ultralytics/cfg/models/yolo-detr/yolov8-detr-AIFI-LPE.yaml

    使用RT-DETR中的TransformerDecoderHead和LearnedPositionalEncoding改进yolov8.(详细介绍请看百度云视频-20231119更新说明)

5. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV2.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积DCNV2改进yolov8.

6. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV3.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积[DCNV3 CVPR2023](https://github.com/OpenGVLab/InternImage)改进yolov8.(安装教程请看百度云视频-20231119更新说明)

7. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV2-Dynamic.yaml

    使用RT-DETR中的TransformerDecoderHead和自研可变形卷积DCNV2-Dynamic改进yolov8.(详细介绍请看百度云视频-MPCA与DCNV2_Dynamic的说明)

8. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Ortho.yaml

    使用RT-DETR中的TransformerDecoderHead和[OrthoNets](https://github.com/hady1011/OrthoNets/tree/main)中的正交通道注意力改进yolov8.(详细介绍请看百度云视频-20231119更新说明)

9. ultralytics/cfg/models/yolo-detr/yolov8-detr-attention.yaml

    添加注意力到基于RTDETR-Head中的yolov8中.(手把手教程请看百度云视频-手把手添加注意力教程)

10. ultralytics/cfg/models/yolo-detr/yolov8-detr-p2.yaml

    添加小目标检测头P2到TransformerDecoderHead中.

11. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DySnake.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C2f融合.  

12. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block改进yolov8.

13. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-Rep.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv二次创新后的Faster-Block-Rep改进yolov8.

14. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block-EMA的Faster-Block-EMA改进yolov8.

15. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Faster-Rep-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv、[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block改进yolov8.

16. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-AKConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进yolov8.

17. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov8.

18. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov8.

19. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCBAMConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov8.

20. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-Conv3XC.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进yolov8.

21. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-SPAB.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的SPAB改进yolov8.

22. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DRB.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进yolov8.

23. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-UniRepLKNetBlock.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的UniRepLKNetBlock改进yolov8.

24. ultralytics/cfg/models/yolo-detr/yolov8-detr-DWR-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)进行二次创新改进yolov8.

25. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DBB.yaml

    使用RT-DETR中的TransformerDecoderHead和[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进yolov8.

26. ultralytics/cfg/models/yolo-detr/yolov8-detr-CSP-EDLAN.yaml

    使用RT-DETR中的TransformerDecoderHead和[DualConv](https://github.com/ChipsGuardian/DualConv)打造CSP Efficient Dual Layer Aggregation Networks改进yolov8.

27. ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF.yaml

    使用RT-DETR中的TransformerDecoderHead和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov8.

28. ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF-P2.yaml

    在ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.

29. ultralytics/cfg/models/yolo-detr/yolov8-detr-slimneck.yaml

    使用RT-DETR中的TransformerDecoderHead和[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中VoVGSCSP\VoVGSCSPC和GSConv改进yolov8的neck.

30. ultralytics/cfg/models/yolo-detr/yolov8-detr-slimneck-asf.yaml

    在ultralytics/cfg/models/yolo-detr/yolov8-detr-slimneck.yaml使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新.

31. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-AggregatedAtt.yaml

    使用RT-DETR中的TransformerDecoderHead和[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进C2f.(百度云视频-20240106更新说明)

32. ultralytics/cfg/models/yolo-detr/yolov8-detr-SDI.yaml

    使用RT-DETR中的TransformerDecoderHead和[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对yolov8中的feature fusion进行改进.

33. ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.

34. ultralytics/cfg/models/yolo-detr/yolov8-detr-goldyolo-asf.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行改进特征融合模块.

35. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进C2f.

36. ultralytics/cfg/models/yolo-detr/yolov8-detr-HSFPN.yaml

    利用RT-DETR中的TransformerDecoderHead和使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进YOLOV8中的PAN.

37. ultralytics/cfg/models/yolo-detr/yolov8-detr-HSPAN.yaml

    利用RT-DETR中的TransformerDecoderHead和对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进YOLOV8中的PAN.

38. ultralytics/cfg/models/yolo-detr/yolov8-detr-Dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov8-detr neck中的上采样.

39. ultralytics/cfg/models/yolo-detr/yolov8-detr-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov8-detr neck中的上采样.

40. ultralytics/cfg/models/yolo-detr/yolov8-detr-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8-detr neck的下采样.

41. ultralytics/cfg/models/yolo-detr/yolov8-detr-ASF-Dynamic.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion的上采样模块得到Dynamic Sample Attentional Scale Sequence Fusion改进yolov8-detr中的neck.

42. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进yolov8-detr中的C2f.

43. ultralytics/cfg/models/yolo-detr/yolov8-detr-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov8-detr中的C2f.

44. ultralytics/cfg/models/yolo-detr/yolov8-detr-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov8-detr中的C2f.

45. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)对C2f中的BottleNeck进行改进,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

46. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-LVMB.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)与Cross Stage Partial进行结合,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

47. ultralytics/cfg/models/yolo-detr/yolov8-detr-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行改进yolov8-detr.

48. ultralytics/cfg/models/yolo-detr/yolov8-detr-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        block模块选择,具体可看对应百度云视频-20240302更新公告.
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

49. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided和Light-weight Context Guided DownSample改进yolov8-detr.

50. ultralytics/cfg/models/yolo-detr/yolov8-detr-PACAPN.yaml

    自研结构, Parallel Atrous Convolution Attention Pyramid Network, PAC-APN

51. ultralytics/cfg/models/yolo-detr/yolov8-detr-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进yolov8-detr.

52. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-RetBlock.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进C2f.

53. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块改进C2f.

54. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进C2f.

55. ultralytics/cfg/models/yolo-detr/yolov8-detr-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.

56. ultralytics/cfg/models/yolo-detr/yolov8-detr-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.

57. ultralytics/cfg/models/yolo-detr/yolov8-detr-C2f-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C2f.

### 以Yolov5为基准模型的改进方案
1. ultralytics/cfg/models/yolo-detr/yolov5-detr.yaml

    使用RT-DETR中的TransformerDecoderHead改进yolov5.

2. ultralytics/cfg/models/yolo-detr/yolov5-detr-DWR.yaml

    使用RT-DETR中的TransformerDecoderHead和[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块改进yolov5.

3. ultralytics/cfg/models/yolo-detr/yolov5-detr-fasternet.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)改进yolov5.(支持替换其他主干,请看百度云视频-替换主干示例教程)

4. ultralytics/cfg/models/yolo-detr/yolov5-detr-AIFI-LPE.yaml

    使用RT-DETR中的TransformerDecoderHead和LearnedPositionalEncoding改进yolov5.(详细介绍请看百度云视频-20231119更新说明)

5. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV2.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积DCNV2改进yolov5.

6. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV3.yaml

    使用RT-DETR中的TransformerDecoderHead和可变形卷积[DCNV3 CVPR2023](https://github.com/OpenGVLab/InternImage)改进yolov5.(安装教程请看百度云视频-20231119更新说明)

7. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV2-Dynamic.yaml

    使用RT-DETR中的TransformerDecoderHead和自研可变形卷积DCNV2-Dynamic改进yolov5.(详细介绍请看百度云视频-MPCA与DCNV2_Dynamic的说明)

8. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Ortho.yaml(详细介绍请看百度云视频-20231119更新说明)

    使用RT-DETR中的TransformerDecoderHead和[OrthoNets](https://github.com/hady1011/OrthoNets/tree/main)中的正交通道注意力改进yolov5.

9. ultralytics/cfg/models/yolo-detr/yolov5-detr-attention.yaml

    添加注意力到基于RTDETR-Head中的yolov5中.(手把手教程请看百度云视频-手把手添加注意力教程)

10. ultralytics/cfg/models/yolo-detr/yolov5-detr-p2.yaml

    添加小目标检测头P2到TransformerDecoderHead中.

11. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DySnake.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C3融合.  

12. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block改进yolov5.

13. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster-Rep.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv二次创新后的Faster-Block-Rep改进yolov5.

14. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block-EMA的Faster-Block-EMA改进yolov5.

15. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Faster-Rep-EMA.yaml

    使用RT-DETR中的TransformerDecoderHead和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中与[RepVGG CVPR2021](https://github.com/DingXiaoH/RepVGG)中的RepConv、[EMA ICASSP2023](https://arxiv.org/abs/2305.13563v1)二次创新后的Faster-Block改进yolov5.

16. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-AKConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进yolov5.

17. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov5.

18. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCAConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov5.

19. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RFAConv.yaml

    使用RT-DETR中的TransformerDecoderHead和[RFCBAMConv 2023](https://github.com/Liuchen1997/RFAConv)改进yolov5.

20. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-Conv3XC.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的Conv3XC改进yolov5.

21. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-SPAB.yaml

    使用RT-DETR中的TransformerDecoderHead和[Swift Parameter-free Attention Network](https://github.com/hongyuanyu/SPAN/tree/main)中的SPAB改进yolov5.

22. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DRB.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进改进yolov5.

23. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-UniRepLKNetBlock.yaml

    使用RT-DETR中的TransformerDecoderHead和[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的UniRepLKNetBlock改进改进yolov5.

24. ultralytics/cfg/models/yolo-detr/yolov5-detr-DWR-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)进行二次创新改进yolov5.

25. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DBB.yaml

    使用RT-DETR中的TransformerDecoderHead和[DiverseBranchBlock CVPR2021](https://github.com/DingXiaoH/DiverseBranchBlock)改进yolov5.

26. ultralytics/cfg/models/yolo-detr/yolov5-detr-CSP-EDLAN.yaml

    使用RT-DETR中的TransformerDecoderHead和[DualConv](https://github.com/ChipsGuardian/DualConv)打造CSP Efficient Dual Layer Aggregation Networks改进yolov5.

27. ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF.yaml

    使用RT-DETR中的TransformerDecoderHead和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolov5.

28. ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF-P2.yaml

    在ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.

29. ultralytics/cfg/models/yolo-detr/yolov5-detr-slimneck.yaml

    使用RT-DETR中的TransformerDecoderHead和[SlimNeck](https://github.com/AlanLi1997/slim-neck-by-gsconv)中VoVGSCSP\VoVGSCSPC和GSConv改进yolov5的neck.

30. ultralytics/cfg/models/yolo-detr/yolov5-detr-slimneck-asf.yaml

    在ultralytics/cfg/models/yolo-detr/yolov5-detr-slimneck.yaml使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新.

31. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-AggregatedAtt.yaml

    使用RT-DETR中的TransformerDecoderHead和[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进C3.(百度云视频-20240106更新说明)

32. ultralytics/cfg/models/yolo-detr/yolov5-detr-SDI.yaml

    使用RT-DETR中的TransformerDecoderHead和[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对yolov5中的feature fusion进行改进.

33. ultralytics/cfg/models/yolo-detr/yolov5-detr-goldyolo.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.

34. ultralytics/cfg/models/yolo-detr/yolov5-detr-goldyolo-asf.yaml

    利用RT-DETR中的TransformerDecoderHead和华为2023最新GOLD-YOLO中的Gatherand-Distribute和[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行改进特征融合模块.

35. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进C3.

36. ultralytics/cfg/models/yolo-detr/yolov5-detr-HSFPN.yaml

    利用RT-DETR中的TransformerDecoderHead和使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进YOLOV5中的PAN.

37. ultralytics/cfg/models/yolo-detr/yolov5-detr-HSPAN.yaml

    利用RT-DETR中的TransformerDecoderHead和对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进YOLOV5中的PAN.

38. ultralytics/cfg/models/yolo-detr/yolov8-detr-Dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolov8-detr neck中的上采样.

39. ultralytics/cfg/models/yolo-detr/yolov8-detr-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolov8-detr neck中的上采样.

40. ultralytics/cfg/models/yolo-detr/yolov8-detr-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolov8-detr neck的下采样.

41. ultralytics/cfg/models/yolo-detr/yolov5-detr-ASF-Dynamic.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion的上采样模块得到Dynamic Sample Attentional Scale Sequence Fusion改进yolov5-detr中的neck.

42. ultralytics/cfg/models/yolo-detr/yolov5-detr-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进yolov5-detr中的C3.

43. ultralytics/cfg/models/yolo-detr/yolov5-detr-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov5-detr中的C2f.

44. ultralytics/cfg/models/yolo-detr/yolov5-detr-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进yolov5-detr中的C2f.

45. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)对C3中的BottleNeck进行改进,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

46. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-LVMB.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)与Cross Stage Partial进行结合,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.

47. ultralytics/cfg/models/yolo-detr/yolov5-detr-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行改进yolov5-detr.

48. ultralytics/cfg/models/yolo-detr/yolov5-detr-bifpn.yaml

    添加BIFPN到yolov8中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        block模块选择,具体可看对应百度云视频-20240302更新公告.
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

49. ultralytics/cfg/models/yolo-detr/yolov5-detr-C2f-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided和Light-weight Context Guided DownSample改进yolov5-detr.

50. ultralytics/cfg/models/yolo-detr/yolov5-detr-PACAPN.yaml

    自研结构, Parallel Atrous Convolution Attention Pyramid Network, PAC-APN

51. ultralytics/cfg/models/yolo-detr/yolov5-detr-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进yolov5-detr.

52. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-RetBlock.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进C3.

53. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块改进C3.

54. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进C3.

55. ultralytics/cfg/models/yolo-detr/yolov5-detr-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.

56. ultralytics/cfg/models/yolo-detr/yolov5-detr-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.

57. ultralytics/cfg/models/yolo-detr/yolov5-detr-C3-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C3.

# 更新公告
- **20231105-rtdetr-v1.0**
    1. 初版项目发布.

- **20231109-rtdetr-v1.1**
    1. 修复断点训练不能正常使用的bug.
    2. 优化get_FPS.py中的模型导入方法.
    3. 增加以yolov5和yolov8为基准模型更换为RTDETR的Head,后续也会提供yolov5-detr,yolov8-detr相关的改进.
    4. 新增百度云视频-20231109更新说明视频和替换主干说明视频.
    5. 新增GhostHGNetV2,RepHGNetV2,详细请看使用教程中的RT-DETR改进方案.
    6. 新增使用DWRSeg中的Dilation-wise Residual(DWR)模块,加强从网络高层的可扩展感受野中提取特征,详细请看使用教程中的RT-DETR改进方案.

- **20231119-rtdetr-v1.2**
    1. 增加DCNV2,DCNV3,DCNV2-Dynamic,并以RTDETR-R18,RTDETR-R50,YOLOV5-Detr,YOLOV8-Detr多个基准模型进行改进,详细请看使用教程中的RT-DETR改进方案.
    2. 使用CVPR2022-OrthoNets中的正交通道注意力改进resnet18-backbone中的BasicBlock,resnet50-backbone中的BottleNeck,yolov8-C2f,yolov5-C3,详细请看使用教程中的RT-DETR改进方案.
    3. 使用LearnedPositionalEncoding改进AIFI中的位置编码信息生成,详细请看使用教程中的RT-DETR改进方案.
    4. 增加EMO模型中的iRMB模块,并使用(EfficientViT-CVPR2023)中的CascadedAttention对其二次创新得到iRMB_Cascaded,详细请看使用教程中的RT-DETR改进方案.
    5. 百度云视频增加1119更新说明和手把手添加注意力机制视频教学.
    6. 更新使用教程.

- **20231126-rtdetr-v1.3**
    1. 支持IoU,GIoU,DIoU,CIoU,EIoU,SIoU.
    2. 支持MPDIoU,Inner-IoU,Inner-MPDIoU.
    3. 支持Normalized Gaussian Wasserstein Distance.
    4. 支持小目标检测层P2.
    5. 支持DySnakeConv.
    6. 新增Pconv,PConv-Rep(二次创新)优化rtdetr-r18与rtdetr-r50.
    7. 新增Faster-Block,Faster-Block-Rep(二次创新),Faster-Block-EMA(二次创新),Faster-Block-Rep-EMA(二次创新)优化rtdetr-r18、rtdetr-r50、yolov5-detr、yolov8-retr.
    8. 更新使用教程.
    9. 百度云视频增加1126更新说明.

- **20231202-rtdetr-v1.4**
    1. 支持AKConv(具有任意采样形状和任意数目参数的卷积核).
    2. 支持RFAConv,RFCAConv,RFCBAMConv(感受野注意力卷积).
    3. 支持UniRepLKNet(大核CNNRepLK正统续作).
    4. 使用CVPR2022 DAttention改进AIFI.
    4. 更新使用教程.
    5. 百度云视频增加1202更新说明.
    6. 解决训练过程中由于指标出现的nan问题导致best.pt没办法正常保存.

- **20231210-rtdetr-v1.5**
    1. 支持来自Swift Parameter-free Attention Network中的重参数化Conv3XC模块.
    2. 支持UniRepLKNet中的DilatedReparamBlock.
    3. 支持UniRepLKNet中的DilatedReparamBlock对DWRSeg中的Dilation-wise Residual(DWR)模块进行二次创新的DWR_DRB.
    4. 使用ICCV2023 FLatten Transformer中的FocusedLinearAttention改进AIFI.
    5. 更新使用教程.
    6. 百度云视频增加1210更新说明.

- **20231214-rtdetr-v1.6**
    1. 支持DiverseBranchBlock.
    2. 利用DualConv打造CSP Efficient Dual Layer Aggregation Networks(仅支持yolov5-detr和yolov8-detr).
    3. 使用Swift Parameter-free Attention Network中的重参数化Conv3XC和DiverseBranchBlock改进RepC3.
    4. 支持最新的ASF-YOLO中的Attentional Scale Sequence Fusion.
    5. 更新使用教程.
    6. 百度云视频增加1214更新说明.

- **20231223-rtdetr-v1.7**
    1. 增加rtdetr-r18-asf-p2.yaml,使用ASF-YOLO中的Attentional Scale Sequence Fusion与Small Object Detection Head进行二次创新.
    2. 新增rtdetr-slimneck.yaml和rtdetr-slimneck-ASF.yaml.
    3. 新增yolov8-detr-slimneck.yaml,yolov8-detr-slimneck-asf.yaml.
    4. 新增yolov5-detr-slimneck.yaml,yolov5-detr-slimneck-asf.yaml.
    5. 修正热力图计算中预处理.
    6. 更新使用教程.
    7. 百度云视频增加1223更新说明.

- **20240106-rtdetr-v1.8**
    1. 新增Shape-IoU,Inner-Shape-IoU.
    2. 新增支持TransNeXt主干和TransNeXt中的聚焦感知注意力机制.
    3. 新增U-NetV2中的Semantics and Detail Infusion Module对RTDETR的CCFM进行创新.
    4. ASF系列支持attention_add.
    5. 更新使用教程.
    6. 百度云视频增加20240106更新说明.

- **20240113-rtdetr-v1.9**
    1. 支持Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    2. 支持Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    3. 支持SlideLoss,EMASlideLoss(利用Exponential Moving Average优化mean iou,可当自研创新模块).
    4. 使用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块.
    5. 使用ASF-YOLO中Attentional Scale Sequence Fusion与GOLD-YOLO中的Gatherand-Distribute进行二次创新结合.
    6. 修正rtdetr-r34中检测头参数错误的问题,增加rtdetr-r34,rtdetr-r50-m的预训练权重.
    7. 更新使用教程.
    8. 百度云视频增加20240113更新说明.

- **20240120-rtdetr-v1.10**
    1. 新增DCNV4.
    2. 使用[LITv2](https://github.com/ziplab/LITv2)中具有提取高低频信息的高效注意力对AIFI进行二次改进.
    3. 使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进RTDETR中的CCFM和YOLOV5-DETR、YOLOV8-DETR中的Neck.
    4. 对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进RTDETR中的CCFM和YOLOV5-DETR、YOLOV8-DETR中的Neck.
    5. 修复没有使用wiou时候断点续寻的bug.
    6. 修复plot_result.py画结果图中乱码的问题.
    7. 更新使用教程.
    8. 百度云视频增加20240120更新说明.

- **20240128-rtdetr-v1.11**
    1. 增加CARAFE轻量化上采样算子.
    2. 增加DySample(ICCV2023)动态上采样算子.
    3. 增加Haar wavelet downsampling下采样算子.
    4. 增加Focaler-IoU,Focaler-GIoU,Focaler-DIoU,Focaler-CIoU,Focaler-EIoU,Focaler-SIoU,Focaler-Shape-IoU,Focaler-MPDIoU.
    5. 增加Focaler-Wise-IoU(v1,v2,v3)(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU).
    6. 使用DySample(ICCV2023)动态上采样算子对ASF-YOLO中的Attentional Scale Sequence Fusion进行二次创新.
    7. 更新使用教程.
    8. 百度云视频增加20240128更新说明.

- **20240206-rtdetr-v1.12**
    1. 新增Shift-ConvNets相关改进内容.(rtdetr-SWC.yaml,rtdetr-R50-SWC.yaml,yolov8-detr-C2f-SWC.yaml,yolov5-detr-C3-SWC.yaml)
    2. 使用UniRepLKNet中的DilatedReparamBlock对EMO中的iRMB进行二次创新.
    3. 使用Shift-ConvNets中的具有移位操作的卷积对EMO中的iRMB进行二次创新.
    4. 更新使用教程.
    5. 百度云视频增加20240206更新说明.

- **20240219-rtdetr-v1.13**
    1. 使用最新的Mamba架构(号称超越Transformer的新架构)改进rtdetr-r18,rtdetr-r50,yolov5-detr,yolov8-detr.
    2. 新增Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2,Wise-Powerful-IoU(v1,v2,v3),Wise-Powerful-IoUV2(v1,v2,v3)系列.
    3. 更新热力图脚本,使用方式可参考最新发的yolov5v7-gradcam的视频.
    4. 更新COCO脚本,增加其他指标输出.
    5. 更新使用教程.
    6. 百度云视频增加20240219更新说明.

- **20240225-rtdetr-v1.14**
    1. 新增YOLOV9中的RepNCSPELAN模块.
    2. 使用DBB,OREPA,DilatedReparamBlock,Conv3XC对YOLOV9中的RepNCSPELAN模块进行二次创新.
    3. 更新使用教程.
    4. 百度云视频增加20240225更新说明.

- **20240302-rtdetr-v1.15**
    1. 新增CGNet中的Light-weight Context Guided和Light-weight Context Guided DownSample模块.
    2. Neck模块新增BIFPN,并对其进行创新,支持替换不同的block.
    3. 为RTDETR定制SlideVarifocalLoss,EMASlideVarifocalLoss.
    4. 更新使用教程.
    5. 百度云视频增加20240302更新说明.

- **20240307-rtdetr-v1.16**
    1. 新增自研Neck结构Parallel Atrous Convolution Attention Pyramid Network, PAC-APN.附带模块内结构图
    2. 复现Lightweight Object Detection中的Dynamic Group Convolution Shuffle Transformer.
    3. 更新使用教程.
    4. 百度云视频增加20240307更新说明.

- **20240321-rtdetr-v1.17**
    1. 新增CVPR2024-RMT主干,并支持RetBlock改进RepC3.
    2. 新增2024年新出的Efficient Local Attention,并用其对HSFPN进行二次创新.
    3. 使用CVPR2021-CoordAttention对HSFPN进行二次创新.
    4. 更新使用教程,增加多个常见疑问解答.
    5. 百度云视频增加20240321更新说明.

- **20240404-rtdetr-v1.18**
    1. 新增CVPR2024 PKINet主干.
    2. 新增CVPR2024 PKINet中的PKIModule和CAA模块,提出C2f-PKI.
    3. 使用CVPR2024 PKINet中的Context Anchor Attention改进RepNCSPELAN、HSFPN.
    4. 新增CVPR2024 Frequency-Adaptive Dilated Convolution.
    5. 增加有效感受野可视化脚本.
    6. 更新使用教程
    7. 百度云视频增加20240404更新说明.

- **20240412-rtdetr-v1.19**
    1. 新增自研Focusing Diffusion Pyramid Network.
    2. 新增HCFNet针对小目标分割的Parallelized Patch-Aware Attention Module改进C2f.
    3. 新增HCFNet针对小目标分割的Dimension-Aware Selective Integration Module对自研Focusing Diffusion Pyramid Network再次进行创新.
    4. 更新使用教程.
    5. 百度云视频增加20240412更新说明.