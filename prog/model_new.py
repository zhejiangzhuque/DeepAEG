import tensorflow.keras.backend as K
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda,MaxPooling1D,Conv1D
from tensorflow.keras.layers import Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.layers.core import Flatten
from layers.graph import GraphConvTest
from tensorflow import keras
import numpy as np
from transformer_drug.new_helper import transformer
import tensorflow as tf
from tensorflow.keras import layers

layers.Layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):

        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(inputs)
        return x + positions
class KerasMultiSourceGCNModel_new(object):

    def __init__(self,use_mut,use_gexp,use_methy,use_copy,regr=True):

        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        self.use_copy = use_copy
        self.regr = regr

    def createMaster(self,drug_dim,edge_dim,smiles_dim,mask_dim,cellline_dim,units_list,unit_edge_list,batch,dropout,activation,use_relu=True,use_bn=True,use_GMP=True):
        nb_genes =cellline_dim[1]
        nb_feats = cellline_dim[2]
        node_feature = Input(batch_shape=(batch,100,drug_dim),name='node_feature')#drug_dim=33 (1,100,33)
        lcq_adj = Input(batch_shape=(batch,100,100,edge_dim),name='lcq_adj') #(1,100,100,11)
        smiles_input=Input(batch_input_shape=(batch,smiles_dim),name="smiles_input")#(1,50)
        mask_input=Input(batch_input_shape=(batch,mask_dim),name="mask")

        copy_input = Input(batch_shape=(batch, nb_genes,), name='copy_input')  # (1, 718)
        mutation_input = Input(batch_shape=(batch, nb_genes,), name='mutation_feat_input')  # (1, 718)
        gexpr_input = Input(batch_shape=(batch, nb_genes,), name='gexpr_feat_input')  # (1, 718)
        methy_input = Input(batch_shape=(batch, nb_genes,), name='methy_feat_input')  # (1, 718)


        # drug feature with GCN
        GCN_layer = GraphConvTest(units=units_list[0],units_edge=unit_edge_list[0],step=0)([node_feature,lcq_adj])
        GCN_layer = [Activation(activation)(item) for item in GCN_layer]
        if use_bn:
            GCN_layer = [BatchNormalization()(item) for item in GCN_layer]
        GCN_layer = [Dropout(dropout)(item) for item in GCN_layer]
        GCN_layer = GraphConvTest(units=units_list[-1],units_edge=unit_edge_list[-1],step=1)(GCN_layer)
        GCN_layer = [Activation(activation)(item) for item in GCN_layer]
        if use_bn:
            GCN_layer = [BatchNormalization()(item) for item in GCN_layer]
        GCN_layer = [Dropout(dropout)(item) for item in GCN_layer]
        #global pooling
        if use_GMP:
            x_drug = GlobalMaxPooling1D()(GCN_layer[0])
            x_drug_edge = GlobalMaxPooling2D()(GCN_layer[1])
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer[0])
            x_drug_edge = GlobalAveragePooling2D()(GCN_layer[1])
        x_drug= Concatenate()([x_drug, x_drug_edge])

        print(x_drug.shape,"x_drug.shape")

        x_smile_transformer=transformer()([smiles_input,mask_input])
        print(x_smile_transformer.shape,"wuhuwuhu")

        print(copy_input.shape,mutation_input.shape,gexpr_input.shape,methy_input.shape)
        x_copy = Dense(256)(copy_input)
        x_copy = Activation('tanh')(x_copy)
        x_copy = BatchNormalization()(x_copy)
        # x_copy = Dropout(0.1)(x_copy)
        x_copy = Dropout(0.2)(x_copy)

        # x_copy = Dense(100, activation=activation)(x_copy)
        x_copy = Dense(100, activation='relu')(x_copy)
        # mutation feature
        x_mutation = Dense(256)(mutation_input)
        x_mutation = Activation('tanh')(x_mutation)
        x_mutation = BatchNormalization()(x_mutation)
        # x_mutation = Dropout(0.1)(x_mutation)
        x_mutation = Dropout(0.2)(x_mutation)
        # x_mutation = Dense(100, activation=activation)(x_mutation)
        x_mutation = Dense(100, activation='relu')(x_mutation)
        # gexpr feature
        x_gexpr = Dense(256)(gexpr_input)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        # x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dropout(0.2)(x_gexpr)
        # x_gexpr = Dense(100, activation=activation)(x_gexpr)
        x_gexpr = Dense(100, activation='relu')(x_gexpr)
        # methylation feature
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        # x_methy = Dropout(0.1)(x_methy)
        x_methy = Dropout(0.2)(x_methy)
        # x_methy = Dense(100, activation=activation)(x_methy)
        x_methy = Dense(100, activation='relu')(x_methy)

        x_gene_res = Concatenate()([x_copy,x_mutation,x_gexpr,x_methy])

        #make (batch ,100 ,4) and attention
        #res=K.stack([x_mutation,x_gexpr,x_methy,x_copy],axis=2)
        #res_att=Flatten()(res)
        #att_vec = Dense(4, activation='softmax', name='attention_vec')(res_att)
        #att_vec=tf.expand_dims(att_vec,axis=1)
        #x_gene = Multiply()([att_vec, res])
        #x_gene_res=tf.reduce_sum(x_gene,axis=2)

        print(x_gene_res.shape,"gene")


# ##############################
#         x_drug= Dense(200)(x_drug)
#         x_smile_transformer=Dense(200)(x_drug)
#         res1=K.stack([x_smile_transformer,x_drug],axis=2)
#         res1_att=Flatten()(res1)
#         att_vec1 = Dense(2, activation='softmax', name='attention_vec1')(res1_att)
#         att_vec1=tf.expand_dims(att_vec1,axis=1)
#         x_drug_final = Multiply()([att_vec1, res1])
#         x_drug_final=tf.reduce_sum(x_drug_final,axis=2)
#         print(x_drug_final.shape,"drug")
# ##############################

        
        x = Concatenate()([x_gene_res,x_drug])
        
        #x = Concatenate()([x_gene_res,x_drug])
        print(x.shape,"final shape")
        # tmp1=Concatenate()(x_gene,x_smile_transformer)
        # tmp2=Concatenate()(x_drug,x_smile_transformer)



        x = Dense(300,activation = 'tanh')(x)
        # x = Dropout(0.1)(x)
        x = Dropout(0.2)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)

        # x = Conv2D(filters=60, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        # x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = "relu",padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = "relu",padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = "relu",padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        # x = Dropout(0.1)(x)   
        x = Dropout(0.2)(x)

        x = Flatten()(x)
        x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        # model = Model(inputs=[node_feature,lcq_adj,gene_feature_input],outputs=output)
        model = Model(inputs=[node_feature,lcq_adj,smiles_input,mask_input,copy_input,mutation_input,gexpr_input,methy_input],outputs=output)

        return model

