import tensorflow as tf
import os
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, mask=None, training=None):

        attn_output = self.att(inputs, inputs,attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

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


class transformer(layers.Layer):
    def __init__(self,**kwargs):
        super(transformer, self).__init__()
        input_dim_drug = 2586
        transformer_emb_size_drug = 128
        transformer_n_layer_drug = 1
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8

        self.emb = TokenAndPositionEmbedding(50, input_dim_drug, transformer_emb_size_drug)
        self.encoder_layer = [TransformerBlock(transformer_emb_size_drug,transformer_num_attention_heads_drug,transformer_intermediate_size_drug)for i in range(transformer_n_layer_drug)]

    def call(self, inputs):
        x1=inputs[0]
        mask=inputs[1]
        ex_mask=K.expand_dims(mask,1)
        ex_mask=K.expand_dims(ex_mask,1)
        # ex_mask = (1.0 - ex_mask) * -10000.0
        #ex_mask = tf.cast(ex_mask, dtype=tf.bool)
        x = self.emb(x1)
        for encoder in self.encoder_layer:
            x=encoder(x,ex_mask)

        return x[:, 0]
