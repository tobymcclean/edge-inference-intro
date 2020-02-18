import argparse
import sys
import os
import logging as log

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

classes_color_map = [
    (255, 255, 255),
    (58, 55, 169),
    (211, 51, 17),
    (157, 80, 44),
    (23, 95, 189),
    (210, 133, 34),
    (76, 226, 202),
    (101, 138, 127),
    (223, 91, 182),
    (80, 128, 113),
    (235, 155, 55),
    (44, 151, 243),
    (159, 80, 170),
    (239, 208, 44),
    (128, 50, 51),
    (82, 141, 193),
    (9, 107, 10),
    (223, 90, 142),
    (50, 248, 83),
    (178, 101, 130),
    (71, 30, 204)
]

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
parser.add_argument('-nt', '--top_n', type=int, required=False, help='Optional. The number of classes to print out.',
                    default=10)
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

# -----------------------------------------------------------------------------
# ----------------- Load image ------------------------------------------------
# -----------------------------------------------------------------------------
log.info('Loading image')
ifile = args['ifile']
images = np.ndarray(shape=(n, c, h, w))
image = cv2.imread(ifile)

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

if len(res.shape) == 3:
    res = np.expand_dims(res, axis=1)

if len(res.shape) == 4:
    _, _, out_h, out_w = res.shape
else:
    log.error(f'Unexpected output blob shape {res.shape}. Only 4D and 3D output blobs are supported.')

# -----------------------------------------------------------------------------
# ----------------- Process results  ------------------------------------------
# -----------------------------------------------------------------------------
log.info('Processing detected objects')

for batch, data in enumerate(res):
    classes_map = np.zeros(shape=(out_h, out_w, 3), dtype=np.int)
    for i in range(out_h):
        for j in range(out_w):
            if len(data[:, i, j]) == 1:
                pixel_class = int(data[:, i, j])
            else:
                pixel_class = np.argmax(data[:, i, j])
            classes_map[i, j, :] = classes_color_map[min(pixel_class, 20)]
    out_img = os.path.join(os.path.dirname(__file__), "out_{}.bmp".format(batch))
    cv2.imwrite(out_img, classes_map)
    classes_map = classes_map.astype(np.uint8)
    image = cv2.imread(ifile)
    classes_map = cv2.resize(classes_map, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    output = ((0.3 * image) + (0.7 * classes_map)).astype('uint8')
    cv2.imshow('OV Segmentation', output)
    cv2.waitKey(0)


# -----------------------------------------------------------------------------
# ----------------- All done --------------------------------------------------
# -----------------------------------------------------------------------------
cv2.destroyAllWindows()
sys.exit(0)