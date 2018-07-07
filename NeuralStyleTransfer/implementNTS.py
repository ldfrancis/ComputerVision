"""
This code was inspired by the deeplearning.ai CNN course's week 4 assignments.

And it uses the implementation provided by http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style

"""

import os
import sys
import time
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
from .utils import cwd, download_if_not_exists


cimage = cwd+"images/louvre_small.jpg"
simage = cwd+"images/monet.jpg"

url = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
file_name = cwd+"pretrained-model/imagenet-vgg-verydeep-19.mat"



# Output folder for the images.
OUTPUT_DIR = 'output/'
# Style image to use.
# STYLE_IMAGE = 'images/starry_night.jpg'
STYLE_IMAGE = simage
# Content image to use.
# CONTENT_IMAGE = 'images/hong_kong_2.jpg'
CONTENT_IMAGE = cimage
# Image dimensions constants.
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
COLOR_CHANNELS = 3

###############################################################################
# Algorithm constants
###############################################################################
# Noise ratio. Percentage of weight of the noise for intermixing with the
# content image.
NOISE_RATIO = 0.6
# Number of iterations to run.
ITERATIONS = 5000
# Constant to put more emphasis on content loss.
BETA = 5
# Constant to put more emphasis on style loss.
ALPHA = 100
# Constant to put more emphasis on the total variation loss.
GAMMA = 100

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, noise_ratio):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(
            -20, 20,
            (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image

def load_image(path):
    image = scipy.misc.imread(path)
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

def save_image(path, image):
    # Output should add back the mean.
    image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def load_vgg_model(path, IMAGE_HEIGHT, IMAGE_WIDTH):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """

    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

        return W, b

    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])

    return graph




def content_loss_func(sess, model):
    """
    Content loss function as defined in the paper.
    """
    def _content_loss(p, x):
        # N is the number of filters (at layer l).
        N = p.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = p.shape[1] * p.shape[2]
        # Interestingly, the paper uses this form instead:
        #
        #   0.5 * tf.reduce_sum(tf.pow(x - p, 2))
        #
        # But this form is very slow in "painting" and thus could be missing
        # out some constants (from what I see in other source code), so I'll
        # replicate the same normalization constant as used in style loss.
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

def style_loss_func(sess, model):
    """
    Style loss function as defined in the paper.
    """
    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        """
        The style loss calculation.
        """
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = _gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    # Layers to use. We will use these layers as advised in the paper.
    # To have softer features, increase the weight of the higher layers
    # (conv5_1) and decrease the weight of the lower layers (conv1_1).
    # To have harder features, decrease the weight of the higher layers
    # (conv5_1) and increase the weight of the lower layers (conv1_1).
    layers = [
        ('conv1_1', 0.5),
        ('conv2_1', 1.0),
        ('conv3_1', 1.5),
        ('conv4_1', 3.0),
        ('conv5_1', 4.0),
    ]

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
    W = [w for _, w in layers]
    loss = sum([W[l] * E[l] for l in range(len(layers))])
    return loss

def tv_loss_func(sess, model):
    x = sess.run(model['input'])
    a = tf.square(x[:, :x.shape[1]-1, :x.shape[2]-1, :] - x[:, 1:, :x.shape[2]-1, :])
    b = tf.square(x[:, :x.shape[1]-1, :x.shape[2]-1, :] - x[:, :x.shape[1]-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def setImageDim(width = 400, height = 300):
    global IMAGE_HEIGHT, IMAGE_WIDTH
    IMAGE_HEIGHT = height
    IMAGE_WIDTH = width


def run(iterations = ITERATIONS, content_image=CONTENT_IMAGE, style_image=STYLE_IMAGE, noise_ratio=NOISE_RATIO):
    with tf.Session() as sess:

        download_if_not_exists(file_name, url)

        # Load the images.
        content_image = load_image(content_image)
        style_image = load_image(style_image)
        # Load the model.
        model = load_vgg_model(cwd+"pretrained-model/imagenet-vgg-verydeep-19.mat", IMAGE_HEIGHT, IMAGE_WIDTH)

        # Generate the white noise and content presentation mixed image
        # which will be the basis for the algorithm to "paint".
        input_image = generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, noise_ratio)

        sess.run(tf.global_variables_initializer())
        # Construct content_loss using content_image.
        sess.run(model['input'].assign(content_image))
        content_loss = content_loss_func(sess, model)

        # Construct style_loss using style_image.
        sess.run(model['input'].assign(style_image))
        style_loss = style_loss_func(sess, model)

        sess.run(model['input'].assign(input_image))
        total_variational_loss = tv_loss_func(sess, model)

        # Instantiate equation 7 of the paper.
        total_loss = BETA * content_loss + ALPHA * style_loss + GAMMA * total_variational_loss

        # From the paper: jointly minimize the distance of a white noise image
        # from the content representation of the photograph in one layer of
        # the neywork and the style representation of the painting in a number
        # of layers of the CNN.
        #
        # The content is built from one layer, while the style is from five
        # layers. Then we minimize the total_loss, which is the equation 7.
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(total_loss)

        sess.run(tf.global_variables_initializer())

        tic = time.time()
        for i in range(iterations):
            sess.run(train_step)
            if (i+1) % 20 == 0:
                Jt, Jc, Js, Jv = sess.run([total_loss, content_loss, style_loss, total_variational_loss])
                print("Iteration " + str(i+1) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                print("total variational loss = ", str(Jv))

                generated_image = sess.run(model['input'])
                # save current generated image in the "/output" directory
                save_image(cwd+"output/" + str(i+1) + ".png", generated_image)

                print("Time elapsed: ", time.time() - tic)
                tic = time.time();
        sess.close()

        # save last generated image
        save_image(cwd+'output/generated_image.jpg', generated_image)
