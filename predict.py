import tensorflow as tf
import numpy as np
import cv2
import math


def create_placeholders(n_H, n_W, n_C, n_y):
    """ Create the placeholders for the tensorflow session
    Arguments:
        n_H scalar int -- height of input images
        n_W scalar int -- width of input images
        n_C scalar int -- number of channels of input
        n_y scalar int -- lenght of output
    Returns:
        X -- placeholder for input data ( input layer )
        Y_1-- placeholder for input labels probability of
              object present in left and rigth( output layer )
        Y_2 -- placeholder for input labels center of balls
    """
    X = tf.placeholder(tf.float32, [None, n_H, n_W, n_C])
    Y = tf.placeholder(tf.float32, [None, 6])
    return X, Y


def initialize_the_parameters():
    W_1 = np.load("W_1.npy")
    W_2 = np.load("W_2.npy")
    W_3 = np.load("W_3.npy")
    W_4 = np.load("W_4.npy")
    W_5 = np.load("W_5.npy")
    W_6 = np.load("W_6.npy")
    W_7 = np.load("W_7.npy")
    b_1 = np.load("b_1.npy")
    b_2 = np.load("b_2.npy")
    Wv_1 = tf.constant(
        value=W_1,
    )
    Wv_2 = tf.constant(
        value=W_2,
    )
    Wv_3 = tf.constant(
        value=W_3,
    )
    Wv_4 = tf.constant(
        value=W_4,
    )
    Wv_5 = tf.constant(
        value=W_5,
    )
    Wv_6 = tf.constant(
        value=W_6,
    )
    Wv_7 = tf.constant(
        value=W_7,
    )
    bv_1 = tf.constant(
        value=b_1,
    )
    bv_2 = tf.constant(
        value=b_2
    )
    parameters = {
        "W_1": Wv_1,
        "W_2": Wv_2,
        "W_3": Wv_3,
        "W_4": Wv_4,
        "W_5": Wv_5,
        "W_6": Wv_6,
        "W_7": Wv_7,
        "b_1": bv_1,
        "b_2": bv_2,
    }
    return parameters


def forward_propagation(X, parameters):
    """Forward propagation for model

    Arguments:
        X tf.placholder -- input data placeholder
        parameters dict -- parameters for model
    """

    W_1 = parameters["W_1"]
    W_2 = parameters["W_2"]
    W_3 = parameters["W_3"]
    W_4 = parameters["W_4"]
    W_5 = parameters["W_5"]
    W_6 = parameters["W_6"]
    W_7 = parameters["W_7"]
    b_1 = parameters['b_1']
    b_2 = parameters["b_2"]
    # CONV2D operation of strides padding valid
    Z_1 = tf.nn.conv2d(X, filter=W_1, strides=[1, 1, 1, 1], padding="VALID")
    # RELU
    A_1 = tf.nn.relu(Z_1)
    # MAXPOOL window 2x2 strides 2x2
    A_1 = tf.nn.max_pool(
     value=A_1,
     ksize=[1, 4, 4, 1],
     strides=[1, 4, 4, 1],
     padding="VALID"
     )
    # CONV2D operation of strides padding valid
    Z_2 = tf.nn.conv2d(
     input=A_1,
     filter=W_2,
     strides=[1, 1, 1, 1],
     padding="VALID"
     )
    # RELU
    A_2 = tf.nn.relu(Z_2)
    # MAXPOOL window 2x2 strides 2x2
    A_2 = tf.nn.max_pool(
     value=A_2,
     ksize=[1, 4, 4, 1],
     strides=[1, 4, 4, 1],
     padding="VALID"
     )
    # CONV2D operation of strides padding valid
    Z_3 = tf.nn.conv2d(
     input=A_2,
     filter=W_3,
     strides=[1, 1, 1, 1],
     padding="VALID")
    # RELU
    A_3 = tf.nn.relu(Z_3)
    # MAXPOOL window 2x2 strides 2x2
    A_3 = tf.nn.max_pool(
     value=A_3,
     ksize=[1, 4, 4, 1],
     strides=[1, 4, 4, 1],
     padding="VALID"
     )
    # CONV2D operation of strides padding valid
    Z_4 = tf.nn.conv2d(
     input=A_3,
     filter=W_4,
     strides=[1, 1, 1, 1],
     padding="VALID",
     )
    # RELU
    A_4 = tf.nn.relu(Z_4)
    # CONV2D operation of strides padding valid
    Z_5 = tf.nn.conv2d(
     input=A_4,
     filter=W_5,
     strides=[1, 1, 1, 1],
     padding="VALID"
     )
    # RELU
    A_5 = tf.nn.relu(Z_5)
    print(A_5.shape)
    # MAXPOOL window 2x2 strides 2x2
    fully_connected = tf.contrib.layers.flatten(A_5)
    print(fully_connected.shape)
    f_c_1 = tf.matmul(fully_connected, W_6) + b_1
    f_c_3 = tf.matmul(f_c_1, W_7) + b_2
    return f_c_3

test = np.array([cv2.imread("downloaded/watch-how-this-adorable-little-p.jpg")])
training_set = np.load("training_image.npy")
labels = np.load("labels.npy")
(m, H, W, C) = training_set.shape
with tf.Session() as sess:
    X, Y = create_placeholders(H, W,C, 6)
    parameters = initialize_the_parameters()
    f_c_3 = forward_propagation(X, parameters)
    n = int(input("Input :"))
    pred = sess.run(f_c_3, {X: [training_set[n]]})
    print(pred.shape)
    pred[0,[0,3]] = 1/(1 + np.exp(-pred[0,[0,3]]))
    print("predicted", pred)
    print("orign", labels[n])
    if pred[0,0] > pred[0,3]:
        centre = (int(math.ceil(pred[0,1])), int(math.ceil(pred[0,2]))) 
    else:
        print("right")
        centre = (int(math.ceil(pred[0,4])), int(math.ceil(pred[0,5])))
        print(centre)
    img = cv2.circle(training_set[n],centre, 10, (0,0,255), -1)
    cv2.imshow("hello", img)
    cv2.waitKey(0)

