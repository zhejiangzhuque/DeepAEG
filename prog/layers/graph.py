from __future__ import print_function
#
# from keras import activations, initializers, constraints
# from keras import regularizers
# from keras.engine import Layer
# from  keras.backend.tensorflow_backend import clip
# import tensorflow as tf
from tensorflow.keras import backend as K
# import numpy as np
# from keras.layers import  Reshape
import tensorflow.keras
import tensorflow as tf
# from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
# from keras.layers import BatchNormalization


class GraphConvTest(tensorflow.keras.layers.Layer):

    def __init__(self,
                 units,
                 units_edge,
                 step,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=False,
                 update_edge=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation= tensorflow.keras.activations.get('tanh'),
                 **kwargs):
        super(GraphConvTest, self).__init__(**kwargs)
        self.units = units
        self.units_edge = units_edge
        self.step = step
        self.use_bias = use_bias
        self.activation = tensorflow.keras.activations.get('tanh')
        self.update_edge = update_edge

        super(GraphConvTest, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'units_edge': self.units_edge,
            'step': self.step,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'update_edge': self.update_edge,
        })
        return config

    def build(self, input_shape):
        feature_dim = input_shape[0][2]
        edge_dim = input_shape[1][-1]
        self.test_W = self.add_weight(
            name = 'w1',
            shape=(feature_dim, feature_dim),
            initializer='random_normal',
            trainable=True
        )
        self.out_linear_W = self.add_weight(
            name = 'w4',
            shape=(feature_dim*edge_dim, self.units),
            initializer='random_normal',
            trainable=True
        )
        if self.update_edge:
            self.adj_linear_W = self.add_weight(
                name = 'w2',
                shape=( self.units*2, self.units_edge),
                initializer='random_normal',
                trainable=True
            )
            self.adj_out_linear_W = self.add_weight(
                name = 'w3',
                shape=(self.units_edge+edge_dim, self.units_edge),
                initializer='random_normal',
                trainable=True
            )
        if self.use_bias:
            self.out_linear_b = self.add_weight(
                name = 'b3',
                shape=(self.units,),
                initializer='random_normal',
                trainable=True
            )
            if self.update_edge:
                self.adj_linear_b = self.add_weight(
                    name = 'b1',
                    shape=(self.units_edge,),
                    initializer='random_normal',
                    trainable=True
                )
                self.adj_out_linear_b = self.add_weight(
                    name = 'b2',
                    shape=(edge_dim,),
                    initializer='random_normal',
                    trainable=True
                )
        super(GraphConvTest, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        res=[input_shape[0][:2]+ (self.units,),input_shape[1][:3]+(self.units_edge,)]
        return res


    def call(self, inputs, **kwargs):
        node_feature, lcq_adj=inputs
        lcq_adj=tf.transpose(lcq_adj,(0,3,1,2))
    
        a0, a1, a2, a3 = lcq_adj.shape  # batch*channel*sha1*sha2 edj 32   11 100  100
        m1, m2, m3 = node_feature.shape  # batch*sha1*emb   node     32    100 33

        support=K.reshape(node_feature,(m1*m2,m3))
        support=K.dot(support,self.test_W)
        support=K.reshape(support,(m1,m2,m3))
        # support = K.dot(node_feature.reshape(m1 * m2, m3), self.test_W).reshape(m1, m2, self.out_features)
        output1=K.reshape(lcq_adj,(a0,a1*a2,a3))
        output1=K.batch_dot(output1,support)
        output1=K.reshape(output1,(a0,a1,a2,m3))
        # output1 = K.dot(lcq_adj.reshape(a0, a1 * a2, a3), support).reshape(a0, a1, a2, self.out_features)
        output2=K.expand_dims(support,1)
        output=output1+output2
        # test=1-K.expand_dims(mask,1)
        # output=output*test
        # res=K.sum(output)
        # assert ((output * (1-mask.unsqueeze(1))).sum() == 0)
        # if self.b is not None:
        #     output = output+self.b
        output=K.permute_dimensions(output,(0,2,3,1))
        output=K.reshape(output,(m1,m2,m3*a1))
        output=self.activation(output)

        # output = self.activation(output.permute(0,     2, 3, 1).reshape(m1, m2, m3 * self.num_head))
        # if self.bn:
        #     output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        #output = output * mask
        # output = self.out_linear(output)
        output = K.dot(output, self.out_linear_W)
        if self.use_bias:
            output += self.out_linear_b
        output = self.activation(output)
        # if self.bn:
        #     output = self.mol_bn2(output.reshape(-1, self.out_features)).reshape(m1, m2, self.out_features)
        #output = output * mask


        if self.update_edge:
            tadj = K.concatenate([K.expand_dims(output,1) * K.ones([1, a2, 1, 1]),
                                K.expand_dims(output,2) * K.ones([1, 1, a3, 1])], 3)
            # tadj = self.relu(self.adj_linear(tadj))
            tadj = K.dot(tadj, self.adj_linear_W)
            if self.use_bias:
                tadj += self.adj_linear_b
            tadj=K.relu(tadj)

            tadj=K.concatenate([tadj,K.permute_dimensions(lcq_adj,(0,2,3,1))],3)

            # tadj = K.concatenate([tadj, lcq_adj.permute(0, 2, 3, 1)], 3)
            _adj = lcq_adj

            # adj = self.relu(self.adj_out_linear(tadj))
            adj = K.dot(tadj, self.adj_out_linear_W)

            if self.use_bias:
                adj += self.adj_out_linear_b
            adj=K.relu(adj)
            #adj = adj * adjmask
            # print(_adj.shape, " ??")
            # adj=K.permute_dimensions(adj,(0,3,1,2,))+_adj
            # print(adj.shape, " ??")
            #
            # adj=tf.transpose(adj,(0,2,3,1))
            return [output, adj]
        else:
            lcq_adj=tf.transpose(lcq_adj,(0,2,3,1))
            return [output, lcq_adj]