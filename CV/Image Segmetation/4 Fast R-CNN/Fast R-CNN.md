# [Fast R-CNN (Object Detection)](https://medium.com/coinmonks/review-fast-r-cnn-object-detection-a82e172e87ba)

[![Sik-Ho Tsang](https://miro.medium.com/fit/c/96/96/1*OxjNUHcLFU8-pp-j8su6pg.jpeg)](https://medium.com/@sh.tsang?source=post_page---------------------------)

[Sik-Ho Tsang](https://medium.com/@sh.tsang?source=post_page---------------------------)

In **this story, Fast Region-based Convolutional Network method (Fast R-CNN) [1] is reviewed. It improves the training and testing speed as well as increasing the detection accuracy.**

1. **Fast R-CNN trains the very deep VGG-16 [2] 9× faster than R-CNN [3], 213× faster at test time**
2. **Higher mAP on PASCAL VOC 2012**
3. **Compared to SPPNet [4], it trains VGG-16 3× faster, tests 10× faster, and is more accurate.**

This is an **2015** **ICCV paper** with **over 3000 citations** when I was writing this story.

### 0. What are covered

1. **The Problems of Prior Arts**
2. **ROI Pooling Layer**
3. **Multi-task Loss**
4. **Some Other Ablation Study**
5. **Comparison with State-of-the-art Results**

### 1. The Problems of Prior Arts

* **Multi-stage Pipeline**: R-CNN and SPPNet first trains the CNN for softmax classifier, then uses the feature vectors for training the bounding box regressor. Thus, **R-CNN and SPPNet are not end-to-end training.**
* **Expensive in Space and Time**: As the **feature vectors are stored in harddisk, occupied hundreds of gigabyte**, for training the bounding box regressor.
* **Slow Object Detection**: **At test-time, R-CNN using VGG-16 needs 47s per image using GPU which is slow.**

Fast R-CNN solves above problems!

### 2. **ROI Pooling Layer**

This is actually **a special case of SPP layer in SPPNet with only one pyramid used**. Below illustrates the example:

<img src="https://miro.medium.com/max/1400/1*aB4gy6i8Zc3BasYaQGDVtg.png" width="800">

### 3. ROI Pooling

Suppose we got the **region proposal (left) with h×w**, and we would like to have an **output (right) of H×W** sizes of output layer after pooling. Then, the **area for each pooling area (middle) = h/H × w/W**.

And in the example above, with **input ROI of 5×7**, and **output of 2×2**, the **area for each pooling area is 2×3 or 3×3** after rounding.

And the maximum value within the pooling window is taken as output value for each grid which is the same idea of conventional max pooling layer.

### 4. Multi-task Loss

Since Fast R-CNN is an end-to-end learning architecture to learn the class of object as well as the associated bounding box position and size, the loss is multi-task loss.

<img src="https://miro.medium.com/max/880/1*YzFseoGKhmDrqagVRJ5_qw.png" width="300">

**Multi-task Loss**

**$L_{cls}$ is the log loss for true class u.**

**$L_{los}$ is the loss for bounding box.**

[u≥1] means it is equal to 1 when u≥1. (u=0 is background class)

**Compared with OverFeat, R-CNN, and SPPNet, Fast R-CNN uses multi-task loss to achieve end-to-end learning.**

<img src="https://miro.medium.com/max/1096/1*67iVyCzqapfB5Nyci_zynw.png" width="500">

**Fast R-CNN 图解**

With mutli-task loss, at the ouput, we have softmax and bounding box regressor as shown at the top right of the figure.

3 Models are evaluated:
**S = AlexNet or CaffeNet**

**M = VGG-like wider version of S**

**L = VGG-16**

<img src="https://miro.medium.com/max/1400/1*i0Fq3zitbotf8mZBURqe7w.png" width="500">

**Multi-task Loss Results**



**With multi-task loss, higher mAP is obtained** compared with stage-wise training, i.e. separate training of softmax and bounding box regressor.

------

# 4. Some Other Ablation Study

## 4.1 Multi Scale Training and Testing

An input image is tested using 5 scales.

<img src="https://miro.medium.com/max/1110/1*awn21lCgHv2hKImdlfXSpg.png" width="300">

**1-Scale vs 5-Scale**

**With 5-scale, higher mAP is obtained for every model with the cost of larger test rate (seconds/image).**

## 4.2 SVM vs Softmax

<img src="https://miro.medium.com/max/1144/1*5VLipfjGezKosdjTADi_wg.png" width="300">

**SVM vs Softmax**

In Fast R-CNN (FRCN), **softmax is better than SVM.**

Also, for SVM, the feature vectors need to be stored for hundreds of gigabyte in harddisk, and become stage-wise training while softmax can achieve end-to-end learning without storing feature vectors into harddisk.

## 4.3 Region Proposals

<img src="https://miro.medium.com/max/1116/1*66N_gcm4o7xAgeTb51HK0g.png" width="300">

**Different Proposal Approaches**

It is found that **increasing number of region proposals does not necessary increase mAP.**

**Spare Set using Selective Search (SS) [5] is already good enough as shown in the figure above (Blue solid line) (**SS [5] is being used in R-CNN.)

**It is still a problem that Fast R-CNN needs region proposals from an external source.**

## 4.4 Truncated SVD for faster detection

One of the bottlenecks of testing time is at FC layers.

**Authors use Singular Vector Decomposition (SVD) to reduce the number of connection in order to decrease the test time.**

**The top 1024 singular values from 25088×4096 matrix in FC6 layer,**and **the top 256 singular values from 4096×4096 matrix in FC7 layer.**

<img src="https://miro.medium.com/max/1400/1*Pcx4x1nUkmF8jiszfO8_nA.png" width="400">

**Large Reduction of Test Time for FC6 and FC7 Layers**

------

# **5. Comparison with State-of-the-art Results**

## 5.1 VOC 2007

<img src="https://miro.medium.com/max/2000/1*t4kHGY-VPUKexDiLC5ObzA.png" width="600">

**VOC 2007 Results**

**Fast R-CNN: 66.9% mAPFast R-CNN with difficult examples removed during training (This is the setting of SPPNet): 68.1% mAPFast R-CNN with external VOC 2012 trained: 70.0% mAP**

## 5.2 VOC 2010

<img src="https://miro.medium.com/max/2000/1*uH3kNBlBlLddtxB7zaVerQ.png" width="600">

**VOC 2010 Results**

Similar to VOC 2007, **Fast R-CNN with external VOC 2007 and 2012 trained is the best with 68.8% mAP.**

## 5.3 VOC 2012

<img src="https://miro.medium.com/max/2000/1*nEiW6axqzWZiXaVmokJF4w.png" width="600">

**VOC 2012 Results**

Similar to VOC 2007, **Fast R-CNN with external VOC 2007 trained is the best with 68.4% mAP.**

## 5.4 Training and Testing Time

<img src="https://miro.medium.com/max/1102/1*boqOFsYjqYP2QJvLs2hA3w.png" width="300">

**Training and Testing Time**

As mentioned, **Fast R-CNN trains the very deep VGG-16 [2] 9× faster than R-CNN [3], 213× faster at test time.**

**Compared to SPPNet [4], it trains VGG-16 3× faster, and tests 10× faster.**

------

# References

1. [2015 ICCV] [Fast R-CNN]
    [Fast R-CNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf?source=post_page---------------------------)
2. [2015 ICLR] [VGGNet]
    [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556?source=post_page---------------------------)
3. [2014 CVPR] [R-CNN]
    [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524?source=post_page---------------------------)
4. [2014 ECCV] [SPPNet]
    [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.699.8052&rep=rep1&type=pdf&source=post_page---------------------------)
5. [2013 IJCV] [Selective Search]
    [Selective Search for Object Recognition](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf?source=post_page---------------------------)