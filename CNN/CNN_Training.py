"""
Version 0：
    CNN Structure:
        Image Normalization: -1:1
        Convolution: [5, 5, 1, 64]
        Max pooling: 2x2
        Convolution: [7, 7, 64, 32]
        Max pooling: 2x2
        Fully connected:
            in: 13 * 22 * 32; out: 1024; activation: ReLU
        Fully connected:
            in: 1024; out: 86; activation: ReLU
        Fully connected:
            in: 86; out: 4; activation: None
"""

import cv2
import numpy as np
import os
import tensorflow as tf

def processing_data(path):
    ReshapeRatio = 0.2
    ImageSet=list()
    LabelSet=list()
    for lists in os.listdir(path):
        picpath = os.path.join(PicDir, lists)
        img = cv2.imread(picpath, 0)
        H_size, W_size = img.shape
        img_crop = img[250:506,:]
        img_crop = (img_crop-127)/255

        location = picpath.split('/')[-1].split('_')[:-1]
        tmp = picpath.split('/')[-1].split('_')[-1].split('.')[0]
        location.append(tmp)
        location = location[0:4]
        Location = np.array([int(location[0])/W_size,int(location[1])/H_size,int(location[2])/W_size,int(location[3])/H_size],dtype=float)

        img_rsz = cv2.resize(img_crop, None, fx=ReshapeRatio, fy=ReshapeRatio, interpolation=cv2.INTER_CUBIC)
        #print(img_rsz.shape)
        ImageSet.append(img_rsz)
        LabelSet.append(Location)

        #cv2.circle(img, (int(location[0]), int(location[1])), 3, (255, 0, 0), -1)
        #cv2.circle(img, (int(location[2]), int(location[3])), 3, (255, 0, 0), -1)
        #cv2.imshow('Image', img_rsz)
        #print(img_rsz.shape)
        #cv2.waitKey(500)

    return np.array(ImageSet), np.array(LabelSet)

def random_batch(ImageSet,LabelSet):
    # Number of images in the training-set.
    num_images = len(ImageSet)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = ImageSet[idx]
    y_batch = LabelSet[idx]

    return x_batch, y_batch

"""
for i in range(len(ImageSet)):
    img = ImageSet[i]
    cv2.circle(img, (int(LabelSet[i][0]*img.shape[1]), int(LabelSet[i][1]*img.shape[0])), 3, (255, 0, 0), -1)
    cv2.circle(img, (int(LabelSet[i][2]*img.shape[1]), int(LabelSet[i][3]*img.shape[0])), 3, (255, 0, 0), -1)

    cv2.imshow('Image', img)
    cv2.waitKey(500)
    print(LabelSet[i])
"""



# --------------- Building TensorFlow Graph ---------------

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

  Hidden_fc1 = 1024
  Hidden_fc2 = 86
  Hidden_fc3 = 4

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 51, 86, 1])
    tf.summary.image("Image", x_image)

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([7, 7, 64, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([13 * 22 * 32, Hidden_fc1])
    b_fc1 = bias_variable([Hidden_fc1])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 13 * 22 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([Hidden_fc1, Hidden_fc2])
    b_fc2 = bias_variable([Hidden_fc2])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([Hidden_fc2, Hidden_fc3])
    b_fc3 = bias_variable([Hidden_fc3])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

if __name__ == '__main__':

    # Import and process data
    PicDir = '/home/zhy/Desktop/MyWorks/Wechat_Jump/CNN/Training_Set'
    ImageSet, LabelSet = processing_data(PicDir)
    train_batch_size = 25


    # Create the model
    x = tf.placeholder(tf.float32, [None, 51, 86])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 4])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        total_loss = tf.losses.mean_squared_error(labels=y_, predictions=y_conv)
        total_loss = tf.reduce_mean(total_loss)
        tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(total_loss)

    #graph_location = tempfile.mkdtemp()
    #print('Saving graph to: %s' % graph_location)
    #train_writer = tf.summary.FileWriter(graph_location)
    #train_writer.add_graph(tf.get_default_graph())

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("Tensorboard_Logs/", sess.graph)
        sess.run(tf.global_variables_initializer())

        # Set the path to save tensorflow model
        save_dir = 'TF_Saved_Model/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = save_dir + 'CNN_prid'

        for i in range(200000):
            x_batch, y_batch = random_batch(ImageSet, LabelSet)
            if i % 100 == 0:
                summary, train_accuracy = sess.run([merged, total_loss], feed_dict={x: x_batch, y_: y_batch, keep_prob: 1})
                writer.add_summary(summary, i)
                print('Step %d, Total_loss with dropout %g' % (i, train_accuracy))

            if i % 5000 == 0:
                saver.save(sess, save_path=save_path, global_step=i)
                print("Saved checkpoint.")
            train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})