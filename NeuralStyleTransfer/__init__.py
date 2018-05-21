"""
This code was inspired by the deeplearning.ai CNN course's week 4 assignments

"""

import tensorflow as tf
import numpy as np

# wieghts used to compute the syle cost using results from these layers
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]



def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
        J_content -- scalar that you compute using equation 1 above.
    """
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, [-1, n_C])
    a_G_unrolled = tf.reshape(a_G, [-1, n_C])
    
    # compute the cost with tensorflow
    J_content = (1/(4*n_C*n_W*n_H))*(tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))))
    
    
    return J_content



def gram_matrix(A):
    """
    Argument:
        A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A, A, transpose_b=True)
    
    return GA




def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # Retrieve dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) 
    a_S = tf.reshape(tf.transpose(a_S), [n_C, -1])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, -1])

    # Computing gram_matrices for both images S and G 
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1/(4*(n_C*n_W*n_H)*(n_C*n_W*n_H)))*(tf.reduce_sum(tf.square(tf.subtract(GS,GG))))
    
    
    return J_style_layer



def compute_style_cost(model, STYLE_LAYERS, sess):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them
    
    Returns: 
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 40, beta = 20):
    """
    Computes the total cost function
    
    Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
        J -- total cost as defined by the formula above.
    """
    
    
    J = alpha*J_content + beta*J_style
    
    
    return J