# Hello Classification with OpenVINO<sup>TM</sup>

Toby McClean 

[✉️ toby.mcclean@adlinktech.com](mailto:toby.mcclean@adlinktech.com)
[🔗 https://www.linkedin.com/in/tobymcclean/](https://www.linkedin.com/in/tobymcclean/)

Based on: [https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md] (https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md)

------

The first computer vision capability we're highlighting in this tutorial is image recognition, using classification networks that have been trained on large datasets to identify scenes and objects.

The application accepts an input image and outputs the probability for each class. Having been trained on the ImageNet ILSVRC dataset of [1000 objects](https://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt), the GoogleNet and RestNet-18 models were automatically downloaded during the build step.

As an example, we provide a Python version of the application.

## Code Explained

Now, we are going to walk through creating a new application from scratch in Python for image classification called `ov-classification.py`. The application will load an abitrary image from disk and classify it using a classification network such as `AlexNet`

### Setting up the Project

You can store the `ov-classification.py` file that we will be creating wherever you want on your device. For simplicity, this guide will create it along with some test images inside a directory under the user's home directory; `~/edge-inference-intro`.

Run the following commands from a terminal to create the directory and files.

```
$ cd ~/
$ mkdir edge-inference-intro
$ cd edge-inference-intro
$ touch ov-classification.py
$ wget https://images.pexels.com/photos/241316/pexels-photo-241316.jpeg
$ wget https://images.pexels.com/photos/39855/lamborghini-brno-racing-car-automobiles-39855.jpeg?cs=srgb&dl=yellow-sports-car-during-day-time-39855.jpg&fm=jpg
$ wget https://images.pexels.com/photos/164654/pexels-photo-164654.jpeg?cs=srgb&dl=orange-mercedes-benz-g63-164654.jpg&fm=jpg
$ wget https://images.pexels.com/photos/220938/pexels-photo-220938.jpeg?cs=srgb&dl=adorable-animal-canine-cute-220938.jpg&fm=jpg
$ wget https://images.pexels.com/photos/46505/swiss-shepherd-dog-dog-pet-portrait-46505.jpeg?cs=srgb&dl=white-long-coated-medium-size-dog-sticking-tongue-out-during-46505.jpg&fm=jpg

```

Some test images are downloaded to the folder with the `wget` commands above.

### Download Pretrained Model

```
cd ~/edge-inference-intro
/opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name alexnet

/opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ./public/alexnet/alexnet.caffemodel
```



Next, we'll add the Python code for the program to the empty source file we created here.

### Imports

Add `import` statements to load the modules we will use for classifying images.

```python
import argparse
import sys
import os
import logging as log

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore
```



| Import                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| argparse                  | Package for parsing the command line                         |
| sys                       | Package with system specific constants and functions         |
| os                        | Package that provides a portable way of using operating system dependent functionality |
| cv2                       | Python binding for OpenCV used to read and write images from disk |
| numpy                     | Package for scientific computing, in this application we use it for working with an image |
| logging                   | Package for flexible event logging including logging levels  |
| openvino.inference_engine | Python binding for OpenVINO which is used for classifying images |

### Parse arguments

Next, add some code to parse the command line arguments supported by the application.  There are two mandatory arguments: the image to be classified (```-i``` or ```--ifile```) and the model to use for classification (```-m``` or ```--model```.)



```python
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-i', '--ifile', type=str, required=True,
                    help='Required. Filename of the image to load and classify')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Required. Path to the model to use for classification. Should end in .xml')
parser.add_argument('-o', '--ofile', type=str, required=False,
                    help='Optional. Filename to write the annotated image to', default=None)
parser.add_argument('-l', '--labels', type=str, required=False,
                    help='Optional. Filename of the class id to label mappings', default=None)
parser.add_argument('-nt', '--top_n', type=int, required=False, help='Optional. The number of classes to print out.',
                    default=10)
parser.add_argument('-d', '--device', type=str, required=False,
                    help='Optional. Specify the target device to infer on: CPU, GPU, MYRIAD or HETERO.', default='CPU')

args = parser.parse_args()
args = vars(args)
```

The application also accepts the following optional arguments: 

* A filename to write an image with the top classification and confidence level (```-o``` or ```--ofile```)
* A filename for a file that maps the class id to a human readable label (```-l``` or ```--labels```)
* The number of classes to output. It will first sort the classes based on confidence level (```-nt``` or ```-top_n```). The default is ```10```.
* The device to use to run the model (```-d``` or ```--device```). The value must be one of ```CPU```, ```GPU```, ```MYRIAD``` or ```HETERO```. The default is ```CPU```.

For example, to run the application

```
python openvino/ov-classification.py -m alexnet.xml -i car1.jpeg
```

### Create the OpenVINO<sup>TM</sup> Inference Engine

The following code will load the provided classification model with OpenVINO<sup>TM</sup>. The [OpenVINO<sup>TM</sup> documentation](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) provides a list of pre-trained models for performing classifications. 

In this article we will continue to use AlexNet which can classify the 1000 different classes from the [ImageNet dataset](http://image-net.org/). The classes include:

* different kinds of fruits and vegetables,
* different species of animals,
* different kinds of vehicles,
* different pieces of furniture,
* etc.

```python
model_xml = args['model']
model_bin = os.path.splitext(model_xml)[0] + '.bin'

ie = IECore()
net = IENetwork(model=model_xml, weights=model_bin)
```



We can then ensure that the device that will execute the model supports all of the layers in the model.

```python
supported_layers = ie.query_network(net, args['device'])
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]

if len(not_supported_layers) != 0:
    log.error('...The following layers are not supported by the device.\n {}'.format(', '.join(not_supported_layers)))

```

The application that we are building only supports models that have a single input and a single output. So again we will verify that this condition is met.

```python
assert len(net.inputs.keys()) == 1, 'The application supports single input topologies.'
assert len(net.outputs) == 1, 'The application supports single output topologies'
```

Finally, we instantiate an executable version of the model.

```python
exec_net = ie.load_network(network=net, device_name=args['device'])
```

### Load an image into memory

We now need to load the image that will be classified and ensure that it is the right size and in the right format. For example, OpenVINO models expect the data layout for an image to be channel, height, and width but images are loaded with a height, width, and channel data layout.

To dynamically resize the input image into the size required by the model we compute the dimensions of the model's input layer.

```python
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
net.batch_size = 1

n, c, h, w = net.inputs[input_blob].shape
```

Now we can use OpenCV to load the image

```python
ifile = args['ifile']
image = cv2.imread(ifile)
```

Resize the image if necessary.

```python
if image.shape[:-1] != (h, w):
    log.info(f'Image {ifile} has been resized from {image.shape[:-1]} to {(h, w)}')
    image = cv2.resize(image, (w, h))
```

Change the data layout of the loaded image

```python
image = image.transpose((2, 0, 1))
```

### Classify the Image

We are ready for the most important part classifying the image. The inference engine expects the image to be included in a 4-dimensional array. The reason for this is sometimes models can process image in batches greater than one.

```python
images = np.ndarray(shape=(n, c, h, w))
images[0] = image
res = exec_net.infer(inputs={input_blob: images})
```

### Process the Results

After the inference engine is executed with the input image, a result is produced. This result contains a list of classes and a confidence level for the class. The confidence level is an indicator of how certain the model is the input image is that class. The class of an image is often the class with the highest confidence level.

```python
res = res[out_blob]
```

For this application the classes are sorted highest to lowest based on confidence level. Then the specified (```-nt``` or ```--top_n```) number of classes is output.

Print the result.

```python
number_top = args['top_n']
if args['labels']:
    with open(args['labels'], 'r') as f:
        labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
else:
    labels_map = None

classid_str = 'class'
classid_str = classid_str + ' ' * (25-len(classid_str))
probability_str = 'confidence'
probability_str = probability_str + ' ' * (15-len(probability_str))

for i, probs in enumerate(res):
    probs = np.squeeze(probs)
    top_ind = np.argsort(probs)[-number_top:][::-1]
    log.info(f'{classid_str}  {probability_str}')
    log.info('{}  {}'.format(
        '-' * len(classid_str),
        '-' * len(probability_str)))

    top_class = labels_map[top_ind[0]] if labels_map else f'{top_ind[0]}'
    top_accuracy = probs[top_ind[0]] * 100

    for id in top_ind:
        det_label = labels_map[id] if labels_map else f'{id}'
        det_label = det_label[0:len(classid_str) - 1]
        label_length = len(det_label)
        space_num_before = 0 #(len(classid_str) - label_length) // 2
        space_num_after = len(classid_str) - (space_num_before + label_length) + 2
        space_num_before_prob = 0 #(len(probability_str) - len(str(probs[id]))) // 2
        log.info('{}{}{}{}{:.7f}'.format(
            ' ' * space_num_before,
            det_label,
            ' ' * space_num_after,
            ' ' * space_num_before_prob,
            probs[id]))
```

## Running the Application

To run the application on an image ```car1.jpeg``` using the AlexNet (```alexnet.xml```) model:

```bash
$  python openvino/ov-classification.py -m alexnet.xml -l imagenet_classes.txt -i car1.jpeg
```

Which outputs:

```
[ INFO ] Creating the argument parser...
[ INFO ] Loading model
[ INFO ] ... model file alexnet.xml
[ INFO ] ... weights file alexnet.bin
[ INFO ] Creating inference engine
[ INFO ] ...Checking that the network can be run on the selected device
[ INFO ] ...Checking that the network has a single input and output
[ INFO ] ...Loading the model
[ INFO ] Getting input information
[ INFO ] Loading image
[ INFO ] Image car1.jpeg has been resized from (2624, 3936) to (227, 227)
[ INFO ] Starting inference in synchronous mode
[ INFO ] Processing the output blob
[ INFO ] Outputting the top n classes
[ INFO ] Top 10 results:
[ INFO ] class                      confidence     
[ INFO ] -------------------------  ---------------
[ INFO ] convertible                0.3641867
[ INFO ] car, sport car             0.2427166
[ INFO ] race car, racing car       0.1876955
[ INFO ] radiator grille            0.1207114
[ INFO ] wheel                      0.0233053
[ INFO ] wagon, station wagon, wa   0.0197702
[ INFO ] hack, taxi, taxicab        0.0113124
[ INFO ] pickup truck               0.0073175
[ INFO ] minivan                    0.0062480
[ INFO ] landrover                  0.0054482
```



## Summary

We have built the ```Hello World``` of ```classification``` using OpenVINO<sup>TM</sup>.

