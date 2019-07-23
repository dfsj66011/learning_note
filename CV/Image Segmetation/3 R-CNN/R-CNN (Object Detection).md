# [R-CNN (Object Detection)](https://medium.com/coinmonks/review-r-cnn-object-detection-b476aba290d1)

[![Sik-Ho Tsang](https://miro.medium.com/fit/c/96/96/1*OxjNUHcLFU8-pp-j8su6pg.jpeg)](https://medium.com/@sh.tsang?source=post_page---------------------------)

[Sik-Ho Tsang](https://medium.com/@sh.tsang?source=post_page---------------------------)Follow

**Region-CNN (R-CNN)** [1] is one of the state-of-the-art **CNN-based deep learning object detection approaches**. Based on this, there are **fast R-CNN** and **faster R-CNN** for faster speed object detection as well as **mask R-CNN** for object instance segmentation. On the other hand, there are also other object detection approaches, such as **YOLO** and **SSD**.

To know deep learning object detection approach well, R-CNN is a must read item. And it is a **2014 CVPR paper with about 6000 citations** at the moment I was writing this story. 

**To have object detection, we need to know the class of object and also the bounding box size and location**.

Conventionally, for each image, there is a **sliding window** to search every position within the image as below. It is a simple solution. However, different objects or even same kind of objects can have **different aspect ratios and sizes** depending on the object size and distance from the camera. And **different image sizes** also affect the effective window size. This process will be **extremely slow** if we use deep learning CNN for image classification at each location.

<img src="https://miro.medium.com/max/1400/1*feg0v9MYMkIDqfa1zWBBzA.png" width="400">

**Illustration of Sliding Window (Left) with Different Aspect Ratios and Sizes (Right)**

1. First, R-CNN uses selective search by [2] to **generate about 2K region proposals**, i.e. bounding boxes for image classification.
2. Then, for each bounding box, image classification is done through CNN.
3. Finally, each bounding box can be refined using regression.

<img src="https://miro.medium.com/max/1400/1*CI8tVwe1QIj1Wknh6ZuLWA.png" width="400">

**R-CNN Flowchart**

------

# What will be covered:

1. Selective Search
2. CNN-based Classification and Scoring
3. Results

------

# 1. Selective Search

<img src="https://miro.medium.com/max/1400/1*NXZoM83IKAM9NZzRTJk1jw.png" width="500">

**Selective Search**

Selective search is proposed by [2].

1. First, color similarities, texture similarities, region size, and region filling are used as **non-object-based segmentation**. Therefore we obtain **many small segmented areas** as shown at the bottom left of the image above.
2. Then, bottom-up approach is used that **small segmented areas are merged together to form larger segmented areas.**
3. Thus, **about 2K** **region proposals (bounding box candidates) are generated** as shown in the image.

------

# 2. CNN-based Classification and Scoring

<img src="https://miro.medium.com/max/1400/1*Sequfmhm-iytuxqBjq3kDg.png" width="400">

**R-CNN Flowchart with More Details**

<img src="https://miro.medium.com/max/1400/1*wzflNwJw9QkjWWvTosXhNw.png" width="500">

**Original AlexNet**

**AlexNet [3] is used to extract the CNN features.**

**For each proposal, a 4096-dimensional feature vector is computed** by forward propagating a mean-subtracted 227×227 RGB image through five convolutional layers and two fully connected layers.

The input has the fixed size of 227×227 while bounding boxes have various shapes and sizes. So, **all pixels in a tight bounding box are warped to 227×227 size.**

**The feature vector is scored by SVM** trained for each class.

For each class, **High IoU (Intersection over Union) overlapping bounding boxes are rejected** since they are bounding the same object.

The **predicted bounding box** **can be further fine-tuned** by another bounding box regressor.

------

# 3. Results

## **3.1 VOC 2010**

<img src="https://miro.medium.com/max/1400/1*CbpKWiVsB-beWNgVGoQ6zg.png" width="500">

**VOC 2010**

R-CNN and R-CNN BB obtain the highest mAP (mean average prediction).

------

## 3.2 ILSVRC 2013

<img src="https://miro.medium.com/max/1400/1*CFjNHMUtq4uBAEbKOzaRbg.png" width="400">

**Some Amazing ILSVRC 2013 Results**

<img src="https://miro.medium.com/max/1400/1*0dNYXOVpiXwjFv0GsWVhGw.png" width="400">

**Some ILSVRC 2013 Results with Some Missing Detections**

<img src="https://miro.medium.com/max/1388/1*gNrrvXcMlcqp8Ueg3j92-g.png" width="400">

**ILSVRC 2013**

R-CNN BB even outperforms OverFeat [4], which is the winner of ILSVRC 2013 localization task!

------

## **3.3 VOC 2007**

<img src="https://miro.medium.com/max/1400/1*FsBzLo1WYxBTs43S2LULFw.png" width="500">

**Some examples with high activations in VOC 2007**

<img src="https://miro.medium.com/max/1400/1*6hXU8VS9uyeWr6zFYuUdrQ.png" width="600">

**VOC 2007**

As you may already know, **the CNN used in R-CNN can be changed to any CNNs used in image classification.**

**When R-CNN BB uses VGG-16 [5] which is a 16-layer VGGNet, mAP is even increased to 66.0%**.

------

If interested, please read also my reviews about AlexNet, VGGNet, and OverFeat. (Links at the bottom)

And I will write more reviews for other state-of-the-art deep learning approaches.

------

# References

1. [2014 CVPR] [R-CNN]
    [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524?source=post_page---------------------------)
2. [2013 IJCV] [Selective Search]
    [Selective Search for Object Recognition](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf?source=post_page---------------------------)
3. [2012 NIPS] [AlexNet]
    [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf?source=post_page---------------------------)
4. [2014 ICLR] [OverFeat]
    [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229?source=post_page---------------------------)
5. [2015 ICLR] [VGGNet]
    [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556?source=post_page---------------------------)