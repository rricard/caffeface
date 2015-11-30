from PIL import Image, ImageDraw
import caffe
import scipy.ndimage
import numpy as np

INPUT_IMAGE = 'in/faces.jpg'

NET_PROTOBUF = "facenet_train_test.prototxt"
COMPILED_MODEL = "facenet_iter_200000.caffemodel"

CURSOR_LENGTH = 36
CURSOR_HEIGHT = 36
THRESHOLD = 0.9

POSITIVE_MATCHES_OUTPUT = 'out/%d-face.pgm'
NEGATIVE_MATCHES_OUTPUT = 'out/%d-noface.pgm'

SCALES = np.arange(1.2, 0.2, -0.2)

caffe.set_mode_cpu()
CAFFE_NET = caffe.Net(NET_PROTOBUF, COMPILED_MODEL, caffe.TEST)

INPUT = np.array(Image.open(INPUT_IMAGE).convert('RGB'))

imwriter_counter = 0

# Tile the image in sub-images
def gen_tiles(input, cursor_length = CURSOR_LENGTH, cursor_height = CURSOR_HEIGHT):
    # Define ranges for tiling
    x_window_offsets = np.arange(0, len(input[0]), cursor_length)
    y_window_offsets = np.arange(0, len(input), cursor_height)
    # Split the image into smaller images
    tiles = map(
        lambda x_offset: map(
            lambda y_offset: {
                'image': input[
                    y_offset:y_offset + cursor_length,
                    x_offset:x_offset + cursor_height
                ],
                'rect': [
                    y_offset,
                    x_offset,
                    y_offset + cursor_length,
                    x_offset + cursor_height
                ]
            },
            y_window_offsets
        ),
        x_window_offsets
    )
    # Concatenate the tiles line by line
    return np.concatenate(tiles)

# Test if a tile contains a face and saves it as a training set
def test_tile(tile, threshold = THRESHOLD, caffe_net = CAFFE_NET, pos_matches_out = POSITIVE_MATCHES_OUTPUT, neg_matches_out = NEGATIVE_MATCHES_OUTPUT):
    # Spread the tile
    image = tile['image']
    rect = tile['rect']
    # Shape a net input
    nn_input = image[np.newaxis, np.newaxis, :, :]
    # Load data into the net
    caffe_net.blobs['data'].reshape(*nn_imput.shape)
    caffe_net.blobs['data'].data[...] = nn_input
    # Test with a simple net forwarding
    out = net.forward()

    # Save images and return the results of the test
    if out['loss'][0][1] > threshold:
        Image.fromarray(image).convert('RGB').save(pos_matches_out % (imwriter_counter))
        imwriter_counter += 1
        return true
    else:
        Image.fromarray(image).convert('RGB').save(neg_matches_out % (imwriter_counter))
        imwriter_counter += 1
        return false

# Find faces in a scaled image, ouputs an array of detected rects
def find_in_scale(scaled_input, cursor_length = CURSOR_LENGTH, cursor_height = CURSOR_HEIGHT, threshold = THRESHOLD, caffe_net = CAFFE_NET):
    # Get tiles
    tiles = gen_tiles(scaled_input, cursor_length, cursor_height)
    # Filter on the caffe output and extract rects
    return map(
        lambda tile: tile['rect'],
        filter(
            lambda tile: test_tile(tile, threshold, caffe_net),
            tiles
        )
    )

# Converts an image to greyscale
def rgb_input_to_greyscale(input = INPUT):
    r_in, g_in, b_in = input[:, :, 0], input[:, :, 1], input[:, :, 2]
    grey_out = r_in * 0.2989 + g_in * 0.5870 + b_in * 0.1140
    return grey_out

# Scales the image, finds faces in it, re-scale the matches to the original size
def scale_find_unscale(original_input = INPUT, scale = 1, cursor_length = CURSOR_LENGTH, cursor_height = CURSOR_HEIGHT, threshold = THRESHOLD, caffe_net = CAFFE_NET):
    # Resize the image
    input = scipy.misc.imresize(original_input, scale)
    # Convert to greyscale
    input = rgb_input_to_greyscale(input)
    # Find the faces
    matches = find_in_scale(input, cursor_height, cursor_length, threshold, caffe_net)
    # Rescale the matches and return them
    return map(lambda rect: map(lambda value: value/scale, rect), matches)

# Find faces in an image at all scales, ouputs an array of detected rects
def find_in_image(input = INPUT, scale = SCALES, cursor_length = CURSOR_LENGTH, cursor_height = CURSOR_HEIGHT, threshold = THRESHOLD, caffe_net = CAFFE_NET):
    # Gather results for all scales
    results = map(lambda scale: scale_find_unscale(input, scale, cursor_height, cursor_length, threshold, caffe_net), scales)
    # Merge results inside a single table
    return np.concatenate(results)

# Trace green rects on the image
def trace_rects(rects, original_input = INPUT):
    # Create an image context
    draw = ImageDraw.Draw(image)
    # Draw on it
    map(lambda rect: draw.rectangle(rect[0], rect[1], rect[2], rect[3], fill=None, outline="green"), rects)

rects = find_in_image()
trace_rects(rects)
INPUT.show()
