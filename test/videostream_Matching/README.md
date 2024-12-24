# 视频流特征点推理与匹配使用指南

## 简介

SuperGlue是Magic Leap在CVPR 2020的研究成果，是一种结合图神经网络和最优匹配层的模型，用于图像特征点匹配。它基于SuperPoint的特征点和描述符进行扩展。

### 特性

提供两种预训练模型：

- 室内模型：基于ScanNet训练

- 室外模型：基于MegaDepth训练

主要功能：

- 图像特征点匹配

- 匹配结果的可视化与评估

### 安装依赖

运行以下命令安装必要的Python库：

pip3 install numpy opencv-python torch matplotlib

### 核心脚本

1.	demo_superglue.py: 实时演示SuperPoint + SuperGlue特征点匹配。

2.	match_pairs.py: 从图像对中读取特征点，执行匹配与评估。

### 快速使用

1. 实时匹配演示

使用默认的USB摄像头运行实时匹配：

./demo.py

键盘快捷键：

	•	n: 选取当前帧为锚点图像
	•	e/r: 增加/减少特征点置信度阈值
	•	d/f: 增加/减少匹配过滤阈值
	•	k: 显示/隐藏特征点
	•	q: 退出

2. 图像目录匹配

对图像序列执行匹配并保存可视化结果：

```
./demo.py --input assets/freiburg_sequence/ --output_dir dump_demo_sequence --resize 320 240 --no_display
```

### 图像对匹配与评估

#### 匹配

读取图像对并输出匹配结果至.npz文件：

```
./match_pairs.py
```

#### 可视化匹配

启用匹配结果可视化：

```
./match_pairs.py --viz
```

### 评估

如果提供了真实相对位姿，可使用以下命令评估匹配结果：

```
./match_pairs.py --eval
```



### 推荐设置
室内场景:

```
./match_pairs.py --resize 640 --superglue indoor --max_keypoints 1024 --nms_radius 4
```

室外场景:

```
./match_pairs.py --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float
```