# [Fast-RCNN模型笔记]([https://medium.com/@chu8989802/fast-rcnn-model-%E7%AD%86%E8%A8%98-ec9e75ac4074](https://medium.com/@chu8989802/fast-rcnn-model-筆記-ec9e75ac4074))

[![清TS](https://miro.medium.com/fit/c/96/96/0*EFPzZ7_flbKsdBb4.)](https://medium.com/@chu8989802?source=post_page---------------------------)

[清TS](https://medium.com/@chu8989802?source=post_page---------------------------)

继续2014年的R-CNN，原作者Ross Girshick推出R-CNN的继承者Fast-RCNN。

快速RCNN在继承RCNN同时，吸取了SPPnet的特点。

# 快速RCNN解决R-CNN方法的三个问题：

1. 测试速度慢
2. 训练速度慢
3. 训练所需空间大

# Fast-RCNN的特点与使用场景：

整体训练框架：

1. 生成建议窗口（Selective Search），Fast-RCNN 用选择性搜索方法生成建议窗口（proposal），每张图片约2000个。
2. Fast-RCNN 把整张图片输入CNN，进行特征提取。
3. Fast-RCNN 把建议窗口映射到CNN的最后一层卷积特征图上。
4. 通过 RoI pooling 层使每个建议窗口生成固定尺寸的特征图。
5. 利用 Softmax Loss 和 Smooth L1 Loss 对分类概率和边框回归（Bounding box）联合训练。

整体探测框架：

1. 生成建议窗口（Selective Search），Fast-RCNN用选择性搜索方法生成建议窗口（proposal），每张图片约2000个。
2. Fast-RCNN 把整张图片输入 CNN，进行特征提取。
3. Fast-RCNN 把建议窗口映射到 CNN 的最后一层卷积特征图上。
4. 通过 RoI pooling 层使每个建议窗口生成固定尺寸的特征图。
5. 利用 Softmax Loss 探测分类概率。
6. 利用 Smooth L1 Loss 探测边框回归。
7. 用边框回归值校正原来的建议窗口，生成预测窗口座标。

快速RCNN提供**三种**预测训练网路模型：

1. 小型网路：CaffeNet
2. 中型网路：VGG_CNN_M_1024
3. 大型网路：VGG16