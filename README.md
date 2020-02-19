# Inference and Computer Vision at the Edge
A getting started guide to deploying machine learning models, deep learning models, and computer vision at the Edge with ADLINK Hardware.

This repository uses the leading inference engines from Intel and NVIDIA to deploy optimized solutions.

# Hello World
Almost every technology provides the proverbial Hello World example to get users familiar. And we will be no different 
except that since we working with AI and Computer Vision technologies our Hello World will be an application for 
each of the most common Computer Vision uses: classification, detection and segmentation.

## Hello Classification
[OpenVINO](openvino/openvino-classifying-images.md)

This builds an image classifier using Python and can be run using any of the popular pre-trained classification
models. The example,
* loads an image specified as a command line argument,
* loads a pre-trained classification model,
* executes the model, and
* produces a list of classes and their confidence.

## Hello Detection
[OpenVINO](openvino/openvino-hello-detection.md)

This builds an object detector using Python and can be run using any of the popular pre-trained object detection
models. The example,

* loads an image specified as a command line argument,
* loads a pre-trained object detection model,
* executes the model, and
* shows the image with the detected objects in a bounding box.

## Hello Segmentation
This builds a semantic segmentation application using Python and can be run using any of the popular pre-trained
segmentation models. The example,
* loads an image specified as a command line argument,
* loads a pre-trained segmentation model,
* executes the model, and
* shows the image with different colored mask for each segment.


