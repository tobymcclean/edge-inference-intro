import argparse
import sys
import os
import logging as log

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

# -----------------------------------------------------------------------------
# ----------------- Configure logging -----------------------------------------
# -----------------------------------------------------------------------------
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

# -----------------------------------------------------------------------------
# ----------------- Parse arguments -------------------------------------------
# -----------------------------------------------------------------------------
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
parser.add_argument('-d', '--device', type=str, required=False,
                    help='Optional. Specify the target device to infer on: CPU, GPU, MYRIAD or HETERO.', default='CPU')
parser.add_argument('-x', '--extension', type=str, required=False,
                    help='Optional. Extension for custom layers.', default=None)

args = parser.parse_args()
args = vars(args)

# -----------------------------------------------------------------------------
# ----------------- Load the model and create the inference engine ------------
# -----------------------------------------------------------------------------
log.info(f'Loading model')

model_xml = args['model']
model_bin = os.path.splitext(model_xml)[0] + '.bin'

log.info(f'... model file {model_xml}')
log.info(f'... weights file {model_bin}')

# -----------------------------------------------------------------------------
# ----------------- Create inference engine -----------------------------------
# -----------------------------------------------------------------------------
log.info('Creating inference engine')
ie = IECore()
net = IENetwork(model=model_xml, weights=model_bin)

if args['extension'] and 'CPU' in args['device']:
    ie.add_extension(args['extension'], 'CPU')

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

# -----------------------------------------------------------------------------
# ----------------- Input layer preparation -----------------------------------
# -----------------------------------------------------------------------------
log.info('Getting input information')
input_blob = next(iter(net.inputs))
net.batch_size = 1
input_name = ''
input_info_name = ''

for input_key in net.inputs:
    if len(net.inputs[input_key].layout) == 4:
        input_name = input_key
        net.inputs[input_key].precision = 'U8'
    elif len(net.inputs[input_key].layout) == 2:
        input_info_name = input_key
        net.inputs[input_key].precision = 'FP32'
        if net.inputs[input_key].shape[1] != 3 and net.inputs[input_key].shape[1] != 6 or net.inputs[input_key].shape[
            0] != 1:
            log.error('Invalid input info. Should be 3 or 6 values length.')

n, c, h, w = net.inputs[input_blob].shape

# -----------------------------------------------------------------------------
# ----------------- Output layer preparation ------------------------------------
# -----------------------------------------------------------------------------
out_blob = next(iter(net.outputs))
output_name = ''
output_info = net.outputs[next(iter(net.outputs.keys()))]

for output_key in net.outputs:
    if net.layers[output_key].type == 'DetectionOutput':
        output_name, output_info = output_key, net.outputs[output_key]

if output_name == '':
    log.error('Can not find a DetectionOutput layer in the topology')

output_dims = output_info.shape
if len(output_dims) != 4:
    log.error('Incorrect output dimensions for SSD model')
max_proposal_count, object_size = output_dims[2], output_dims[3]

if object_size != 7:
    log.error('Output item should have 7 as a last dimension')

output_info.precision = 'FP32'

# -----------------------------------------------------------------------------
# ----------------- Load image ------------------------------------------------
# -----------------------------------------------------------------------------
log.info('Loading image')
ifile = args['ifile']
images = np.ndarray(shape=(n, c, h, w))
images_hw = []

image = cv2.imread(ifile)
ih, iw = image.shape[:-1]
images_hw.append((ih, iw))

if image.shape[:-1] != (h, w):
    log.info(f'Image {ifile} has been resized from {image.shape[:-1]} to {(h, w)}')
    image = cv2.resize(image, (w, h))

image = image.transpose((2, 0, 1)) # Change data layout from HWC to CHW
images[0] = image

# -----------------------------------------------------------------------------
# ----------------- Run inference  --------------------------------------------
# -----------------------------------------------------------------------------
log.info('Starting inference in synchronous mode')
res = exec_net.infer(inputs={input_blob: images})

# -----------------------------------------------------------------------------
# ----------------- Get results  ----------------------------------------------
# -----------------------------------------------------------------------------
log.info('Processing the output blob')
res = res[out_blob]

# -----------------------------------------------------------------------------
# ----------------- Process results  ------------------------------------------
# -----------------------------------------------------------------------------
log.info('Processing detected objects')
boxes = {}
classes = {}
data = res[0][0]

for number, proposal in enumerate(data):
    if proposal[2] > 0:
        imid = np.int(proposal[0])
        ih, iw = images_hw[imid]
        label = np.int(proposal[1])
        confidence = proposal[2]
        xmin = np.int(iw * proposal[3])
        ymin = np.int(ih * proposal[4])
        xmax = np.int(iw * proposal[5])
        ymax = np.int(ih * proposal[6])
        print(f'[{number}, {label}] element, prob = {confidence:.6}     ({xmin}, {ymin})-({xmax}, {ymax}) batch id : {imid}', end="\n")
        if confidence > 0.5:
            if not imid in boxes.keys():
                boxes[imid] = []
            boxes[imid].append([xmin, ymin, xmax, ymax])
            if not imid in classes.keys():
                classes[imid] = []
            classes[imid].append(label)

if args['labels']:
    with open(args['labels'], 'r') as f:
        labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
else:
    labels_map = None

image = cv2.imread(ifile)
for imid in classes.keys():
    for idx, box in enumerate(boxes[imid]):
        class_id = classes[imid][idx] - 1
        label = labels_map[class_id] if labels_map else class_id
        image = cv2.putText(image, f'{label}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (232, 35, 244), 2)
        cv2.imwrite('out.jpeg', image)

cv2.imshow('OV Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# ----------------- All done --------------------------------------------------
# -----------------------------------------------------------------------------
sys.exit(0)