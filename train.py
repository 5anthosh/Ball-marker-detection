import tensorflow as tf 
import numpy as np 
import math

training_set = np.load("training_image.npy")
labels = np.load("labels.npy")

def create_placeholders(n_H, n_W, n_C, n_y):
    """ Create the placeholders for the tensorflow session
    
    Arguments:
        n_H scalar int -- height of input images
        n_W scalar int -- width of input images
        n_C scalar int -- number of channels of input
        n_y scalar int -- lenght of output
    Returns:
        X -- placeholder for input data ( input layer )
        Y_1-- placeholder for input labels probability of object present in left and rigth( output layer )
        Y_2 -- placeholder for input labels center of balls 
    """
    X = tf.placeholder(tf.float32, [None,n_H,n_W,n_C])
    Y_1 = tf.placeholder(tf.float32, [None,2])
    Y_2 = tf.placeholder(tf.float32, [None,4])
    return X,Y_1,Y_2

def initialise_the_parameters():
    """Initializes weight parameters 
    Return:
    parameters -- Dictionery containing weight parameters
    """
    W_1 = tf.get_variable(
          name="W1",
          shape=[5,5,3,6],
          initializer=tf.contrib.layers.xavier_initializer(seed=0),
    )
    W_2 = tf.get_variable(
          name="W2",
          shape=[5,5,6,12],
          initializer=tf.contrib.layers.xavier_initializer(seed=0),
    )
    W_3 = tf.get_variable(
          name="W3",
          shape=[5,5,12,24],
          initializer=tf.contrib.layers.xavier_initializer(seed=0),
    )
    W_4 = tf.get_variable(
          name="W4",
          shape=[5,5,24,48],
          initializer=tf.contrib.layers.xavier_initializer(seed=0),
    )
    W_5 = tf.get_variable(
          name="W5",
          shape=[5,5,48,48],
          initializer=tf.contrib.layers.xavier_initializer(seed=0),
    )
    parameters = {
        "W_1":W_1,
        "W_2":W_2,
        "W_3":W_3,
        "W_4":W_4,
        "W_5":W_5,
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
    #CONV2D operation of strides padding valid
    Z_1 = tf.nn.conv2d(input=X, filter=W_1, strides=[1,1,1,1], padding="VALID")
    #RELU 
    A_1 = tf.nn.relu(Z_1)
    #MAXPOOL window 2x2 strides 2x2
    A_1 = tf.nn.max_pool(value=A_1,ksize=[1,2,2,1],strides=[1,2,2,1], padding="VALID")
    #CONV2D operation of strides padding valid
    Z_2 = tf.nn.conv2d(input=A_1, filter=W_2, strides=[1,1,1,1], padding="VALID")
    #RELU 
    A_2 = tf.nn.relu(Z_2)
    #MAXPOOL window 2x2 strides 2x2
    A_2 = tf.nn.max_pool(value=A_2,ksize=[1,2,2,1],strides=[1,2,2,1], padding="VALID")
    #CONV2D operation of strides padding valid
    Z_3 = tf.nn.conv2d(input=A_2, filter=W_3, strides=[1,1,1,1], padding="VALID")
    #RELU 
    A_3 = tf.nn.relu(Z_3)
    #MAXPOOL window 2x2 strides 2x2
    A_3 = tf.nn.max_pool(value=A_3,ksize=[1,2,2,1],strides=[1,2,2,1], padding="VALID")
    #CONV2D operation of strides padding valid
    Z_4 = tf.nn.conv2d(input=A_3, filter=W_4, strides=[1,1,1,1], padding="VALID")
    #RELU 
    A_4 = tf.nn.relu(Z_4)
    #CONV2D operation of strides padding valid
    Z_5 = tf.nn.conv2d(input=A_4, filter=W_5, strides=[1,1,1,1], padding="VALID")
    #RELU 
    A_5 = tf.nn.relu(Z_5)
    #MAXPOOL window 2x2 strides 2x2
    fully_connected = tf.contrib.layers.flatten(A_5)
    f_c_1 = tf.contrib.layers.fully_connected(fully_connected, 600)
    f_c_2 = tf.contrib.layers.fully_connected(f_c_1, 100)
    f_c_3 = tf.contrib.layers.fully_connected(f_c_2, 40)
    output = tf.contrib.layers.fully_connected(f_c_3, 6)
    return output


def compute_cost(f_c_3, Y_1, Y_2):
    """Compute cost
    
    Arguments:
        f_c_3 tensor -- output of forward propagation
        Y numpy.ndarray -- desired output
    Result:
        cost tensor cost function
    """

    cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=Y_1,
            logits=tf.transpose(
                tf.gather(
                    tf.transpose(
                        f_c_3,
                    ),
                    [0,3],
                )
            )
        )
    )  + tf.reduce_mean(
        tf.losses.mean_squared_error(
            labels=Y_2,
            predictions=tf.transpose(
                tf.gather(
                    tf.transpose(
                        f_c_3,
                    ),
                    [1,2,4,5],
                )
            )
        )
    )
    return cost

def random_minibatches(X, Y, n, seed=0): # function for splitting into minibatches
    """creating minibatchs for a epoch
    
    Arguments:
        X numpy.ndarray -- training feature
        Y numpy.ndarray -- labels
        n int -- mini-batch size
    
    Keyword Arguments:
        seed int -- seed for sampling (default: 0)
    
    Returns:
        mini_batches list -- list for mini batches for a epoch
        num_mini_batches int -- number of minibatches in a epoch
    """

    m = X.shape[0] # getting number of training examples
    num_mini_batches = int(math.floor(m/n))
    np.random.seed(seed)
    permu = list(np.random.permutation(np.arange(0, m)))
    shuffled_x = X[permu]
    shuffled_y = Y[permu]
    mini_batchs = []
    for i in range(int(num_mini_batches)):
        mini_x = shuffled_x[i*n:(i+1)*n]
        mini_y = shuffled_y[i*n:(i+1)*n]
        mini_batchs.append((mini_x, mini_y))
    if m%n != 0:
        mini_x = shuffled_x[num_mini_batches*n:]
        mini_y = shuffled_y[num_mini_batches*n:]
        mini_batchs.append((mini_x, mini_y))
        num_mini_batches += 1
    return mini_batchs, num_mini_batches

######## Model ##########
learning_rate = 0.009
num_of_epochs = 100
output_shape = 6
batch_size = 8
(m, H, W, C) = training_set.shape

X, Y_1, Y_2 = create_placeholders(H,W,C,output_shape)
parameters = initialise_the_parameters()
f_c_3 = forward_propagation(X, parameters)
cost = compute_cost(f_c_3, Y_1, Y_2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()
seed = 0
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_of_epochs):
        mini_batches, m = random_minibatches(training_set, labels, batch_size, seed) 
        epoch_cost = 0
        seed += 1
        for j in range(m):
            batch_x, batch_y = mini_batches[j]
            sess.run(optimizer, {X:batch_x, Y_1:batch_y[:,[0,3]], Y_2:batch_y[:, [1,2,4,5]]})
            cost_v = sess.run(cost, {X:batch_x, Y_1:batch_y[:, [0,3]], Y_2:batch_y[:, [1,2,4,5]] })
            epoch_cost += cost_v/m
        print("number " + str(i) + ": " + str(epoch_cost))
    saver = tf.train.Saver()
    saver.save(sess, "my_model")
    np.save(sess.run(W_1), "W_1")
    np.save(sess.run(W_2), "W_2")
    np.save(sess.run(W_3), "W_3")
    np.save(sess.run(W_4), "W_4")
    np.save(sess.run(W_5), "W_5")















