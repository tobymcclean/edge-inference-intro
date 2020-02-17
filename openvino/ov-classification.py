import argparse
import sys
import os
import logging as log

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

# Configure logging
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

# Parse arguments
log.info('Creating the argument parser...')
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

# Load the model and create the inference engine
log.info(f'Loading model')

model_xml = args['model']
model_bin = os.path.splitext(model_xml)[0] + '.bin'

log.info(f'... model file {model_xml}')
log.info(f'... weights file {model_bin}')

log.info('Creating inference engine')
ie = IECore()
net = IENetwork(model=model_xml, weights=model_bin)

log.info('...Checking that the network can be run on the selected device')
supported_layers = ie.query_network(net, args['device'])
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]

if len(not_supported_layers) != 0:
    log.error('...The following layers are not supported by the device.\n {}'.format(', '.join(not_supported_layers)))


log.info('...Checking that the network has a single input and output')
assert len(net.inputs.keys()) == 1, 'The application supports single input topologies.'
assert len(net.outputs) == 1, 'The application supports single output topologies'

log.info('...Loading the model')
exec_net = ie.load_network(network=net, device_name=args['device'])

log.info('Getting input information')
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
net.batch_size = 1

n, c, h, w = net.inputs[input_blob].shape

log.info('Loading image')
ifile = args['ifile']
image = cv2.imread(ifile)

if image.shape[:-1] != (h, w):
    log.info(f'Image {ifile} has been resized from {image.shape[:-1]} to {(h, w)}')
    image = cv2.resize(image, (w, h))

image = image.transpose((2, 0, 1)) # Change data layout from HWC to CHW

log.info('Starting inference in synchronous mode')
images = np.ndarray(shape=(n, c, h, w))
images[0] = image
res = exec_net.infer(inputs={input_blob: images})

log.info('Processing the output blob')
res = res[out_blob]

log.info('Outputting the top n classes')
number_top = args['top_n']
log.info(f'Top {number_top} results:')
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
