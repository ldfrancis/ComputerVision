"""
This code was inspired by the deeplearning.ai CNN course's week 4 assignments

"""

import tensorflow as tf
import numpy as np

# wieghts used to compute the syle cost using results from these layers
STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0)]



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
    m, n_H, n_W, n_C = a_C.get_shape().as_list()
    
    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, [-1, n_C])
    a_G_unrolled = tf.reshape(a_G, [-1, n_C])
    
    # compute the cost with tensorflow
    J_content = (1/(4*n_C*n_W*n_H))*(tf.reduce_sum(tf.pow(a_C-a_G, 2)))
    
    
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

       
        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(model[layer_name])

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = model[layer_name]
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 5, beta = 100):
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