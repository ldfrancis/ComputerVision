import os
import sys
import scipy.io
import scipy.misc
from .utils import *   
from NeuralStyleTransfer import *

import tensorflow as tf
import numpy as np


# Reset the graph
tf.reset_default_graph()

cwd = os.path.realpath(__file__)
cwd = cwd.split('/')[:-1]
cwd = '/'.join(cwd) + '/'
print(cwd)

url = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
file_name = "pretrained-model/imagenet-vgg-verydeep-19.mat"
download_file_if_not_exist(file_name, url)

# Start interactive session
sess = tf.InteractiveSession()

# load, reshape, and normalize our "content" image (the Louvre museum picture):
content_image = scipy.misc.imread(cwd+"images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)

#  load, reshape and normalize our "style" image (Claude Monet's painting):
style_image = scipy.misc.imread(cwd+"images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

# instantiate the generated image as noise
generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

# load the vgg19 model
model = load_vgg_model(cwd+"pretrained-model/imagenet-vgg-verydeep-19.mat")


# compute the content cost:

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)


# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

# Total cost
J = total_cost(J_content, J_style)

# define optimizer 
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step 
train_step = optimizer.minimize(J)


def model_nn(sess = sess, input_image = generated_image, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    model['input'].assign(input_image)
    ### END CODE HERE ###
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        sess.run(train_step)
        ### END CODE HERE ###
        
        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image(cwd+"output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image(cwd+'output/generated_image.jpg', generated_image)
    
    return generated_image