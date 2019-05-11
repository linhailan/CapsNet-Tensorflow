import os
import scipy
import numpy as np
import tensorflow as tf


def load_mnist(batch_size, is_training=True,
               train_data_number=55000, validation_data_number=5000, test_data_number=10000):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        alltrainx = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        alltrainy = loaded[8:].reshape((60000)).astype(np.int32)

        # trainx = alltrainx[:55000] / 255.
        # trainy = alltrainy[:55000]
        trainx = alltrainx[:train_data_number] / 255.
        trainy = alltrainy[:train_data_number]

        # validationx = alltrainx[55000:, ] / 255.
        # validationy = alltrainy[55000:]
        validationx = alltrainx[55000:validation_data_number+55000] / 255.
        validationy = alltrainy[55000:validation_data_number+55000]

        # num_train_batch = 55000 // batch_size
        # num_validation_batch = 5000 // batch_size
        num_train_batch = train_data_number // batch_size
        num_validation_batch = validation_data_number // batch_size

        return trainx, trainy, num_train_batch, validationx, validationy, num_validation_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        teX = testX[:test_data_number] / 255.

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        testy = loaded[8:].reshape((10000)).astype(np.int32)

        teY = testy[:test_data_number]

        # num_te_batch = 10000 // batch_size
        # return teX / 255., teY, num_te_batch
        num_test_batch = test_data_number // batch_size
        return teX, teY, num_test_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False,
              train_data_number=55000, validation_data_number=5000, test_data_number=10000):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training,train_data_number=train_data_number,
                        validation_data_number=validation_data_number,
                        test_data_number=test_data_number)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads,
                   train_data_number=55000, validation_data_number=5000, test_data_number=10000):

    if dataset == 'mnist':
        trX, trY, _, _, _, _ = load_mnist(batch_size, is_training=True, train_data_number=train_data_number,
                                          validation_data_number=validation_data_number,
                                          test_data_number=test_data_number)
    elif dataset == 'fashion-mnist':
        trX, trY, _, _, _, _  = load_fashion_mnist(batch_size, is_training=True)
    else:
        print("dataset error,should be in [mnist,fashin-mnist]")

    # 数据队列？第一个tensorflow操作 ，获取样本切片并存入队列中？
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return X, Y


def save_images(imgs, size, path):
    """
    :param imgs:  [batch_size, image_height, image_width]
    :param size:  a list with tow int elements, [image_height, image_width]
    :param path:  the path to save images
    :return:
    """
    imgs = (imgs + 1.) / 2  # inverse_transform
    return scipy.misc.imsave(path, mergeimgs(imgs, size))


def mergeimgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return shape
