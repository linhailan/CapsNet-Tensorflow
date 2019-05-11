"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import numpy as np
import tensorflow as tf

from config import cfg
from utils import reduce_sum
from utils import softmax
from utils import get_shape


epsilon = 1e-9


class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        """
        :param num_outputs: the number of capsule in this layer.
        :param vec_len: integer, the length of the output vector of a capsule.
        :param with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.
        :param layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        """
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self,capsnet_input, kernel_size=None, stride=None):
        """
        :param capsnet_input:  A 4-D tensor.
        :param kernel_size: will be used while 'layer_type" equal 'CONV'
        :param stride:  will be used while 'layer_type" equal 'CONV'
        :return: A 4-D tensor.
        """
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # the PrimaryCaps layer, a convolutional layer
                # input: [batch_size, 20, 20, 256]
                assert capsnet_input.get_shape() == [cfg.batch_size,20,20,256]

                # NOTE: I can't find out any words from the paper whether the
                # PrimaryCap convolution does a ReLU activation or not before
                # squashing function, but experiment show that using ReLU get a
                # higher test accuracy. So, which one to use will be your choice
                capsules = tf.contrib.layers.conv2d(capsnet_input,self.num_outputs * self.vec_len,
                                                    self.kernel_size,self.stride,padding="VALID",
                                                    activation_fn=tf.nn.relu)
                # capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                #                                    self.kernel_size, self.stride,padding="VALID",
                #                                    activation_fn=None)
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

                # return tensor with shape [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                return(capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(capsnet_input,shape=(cfg.batch_size,-1,1,capsnet_input.shape[-2].value,1))

                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    b_IJ = tf.constant(np.zeros([cfg.batch_size,
                                                 capsnet_input.shape[1].value,
                                                 self.num_outputs, 1, 1],dtype=np.float32))
                    capsules = routing(self.input, b_IJ, num_outputs=self.num_outputs, num_dims=self.vec_len)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(l_input, b_IJ, num_outputs=10, num_dims=16):
    """
    :param l_input:  A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1] shape,
                    num_caps_l是前一层输出的capsule的数量
    :param b_IJ:   A Tensor whth [batch_size,num_caps_l,num_caps_l_plus_1,1,1] shape,
                    代表两层的capsule的关系，是不是向量的方向？
    :param num_outputs: 本层输出的capsule的数量
    :param num_dims:    capsule的维度
    :return:
            A Tensor of shape [batch_size, num_caps_l，num_caps_l_plus_1, length(v_j)=16, 1]
            representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l
        v_j represents the vector output of capsule j in the layer l+1.

        矩阵相乘操作tf.matmul比较耗费时间，可以用一系列操作代替。[a,b]@[b,c]等同于以下操作：
        (1)[a,b]--->[a*c,b],用np.tile或tp.tile实现
        (2)[b,c]--->[b,c*a]--->转置成[c*a,b]
        (3)[a*c,b]*[c*a,b]
        (4)reduce_sum at axis = 1
        (5) reshape to [a,c]
    """
    input_shape = get_shape(l_input)
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))
    l_input = tf.tile(l_input,[1,1,num_dims * num_outputs,1,1])
    """
    W的形状是[1,1152,160,8,1],代表它要表达每张图片1152个输入capsule与160个输出capsule的向量值的关系
    input的形状是[128,1152,1,8,1],代表的是128张图片，每张图片输出1152个capsule,每个capsule的维数的长度是8
        input记录第l层的每个capsule的具体取值
    u_hat的形状是[128,1152,160,1,1]或者[128,1152,10,16,1],
        代表128张图片，每张图片中，第l层的每个capsule对应第l+1层的capsule的向量值,只记录第l层的capsule的个数，不记录取值
    """
    u_hat = reduce_sum(W * l_input,axis=3,keepdims=True)
    assert u_hat.get_shape() == [128,1152,160,1,1]

    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    assert u_hat.get_shape() == [128,1152,10,16,1]

    # assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
