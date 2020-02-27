# Weather Image Recognition Considering Light Condition Via CNNs #
# Project Summary
 In this project, a image classification system based on convolutional neural network (CNN) is applied, which can identify five weather types: sunny, cloudy, rainy, foggy and snowy, and judge the ambient brightness (Bright & Dark) of all kinds of weather.Then, we used a dataset consisting of 8,890 weather images, including 7,899 training images and 991 test images.
## CNNs We have used
 > * BPNN
 > * AlexNet
 > * GoogLeNet+Inception v3
 > * SENet+Inception v3

## Weather dataset
 The weather Dataset is available in this Link https://pan.baidu.com/s/1d9N89qkMuqMaydcXpoALog
 called *'WeatheDataset'*
pls contact me with email for more info || Jasonmils@whut.edu.cn

## Experiment & Result
### 1. Data Augment
 First， the data is resized and fliped for data augment
 ```python
 for i in [-1, 0, 1]:
     file_new = cv2.flip(file, i)
     im = cv2.resize(file, (width, height))
     train_images.append(im.reshape(1, width, height, 3) / 255.0)
     train_labels.append(label)
 ```
### 2. Training & Testing
 Resuls are as follow.

 ![Fig.1 validation Acc vs Epoch](https://github.com/Jasonmils/Weather-Image-Recognition-Considering-Light-Condition-Via-CNNs/blob/master/Result%20(1).png)

 ![Fig.2 validation Loss vs Epoch](https://github.com/Jasonmils/Weather-Image-Recognition-Considering-Light-Condition-Via-CNNs/blob/master/Result%20(2).png)
 
