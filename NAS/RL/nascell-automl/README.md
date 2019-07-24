# [nascell-automl](https://github.com/wallarm/nascell-automl)

> 最新版本代码请参照该项目的源地址：https://github.com/wallarm/nascell-automl

This code belongs to the "Simple implementation of Neural Architecture Search with Reinforcement Learning
" blog post.

An original blog post with all the details (step-by-step guide):
https://lab.wallarm.com/the-first-step-by-step-guide-for-implementing-neural-architecture-search-with-reinforcement-99ade71b3d28

# Requirements
- Python 3
- Tensorflow > 1.4

# Training
Print parameters:
```
python3 train.py --help
```
```
optional arguments:
  -h, --help            show this help message and exit
  --max_layers MAX_LAYERS
```
Train:
```
python3 train.py
```

# For evaluate architecture
Print parameters:
```
$ cd experiments/
$ python3 train.py --architecture "61, 24, 60,  5, 57, 55, 59, 3"
```

# 小结
* 核心原理是借助网络控制器（实际是个RNN）生成 CNN 的网络参数（filter的大小、filter的数量、pool的大小、dropout_rate）
* 默认情况下，网络控制器每给出一个 cnn_config，CNN 训练 500 轮，
* reware = 0.01 if acc - pre_acc > 0.01 else acc
* reware 可以作为 RL 梯度更新过程中的学习率？
* RL 基于梯度方法更新，损失函数计算过程中的 label 是输入的 states ？