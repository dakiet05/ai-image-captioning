import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, RepeatVector

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha_layers = [tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
                           for _ in range(num_layers)]
        self.ffn_layers = [tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ]) for _ in range(num_layers)]
        self.layernorm1_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                                  for _ in range(num_layers)]
        self.layernorm2_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                                  for _ in range(num_layers)]
        self.dropout_layers = [Dropout(rate) for _ in range(num_layers)]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return cfg

    def call(self, inputs, training=None):
        # inputs = [x, enc_output]
        x, enc_output = inputs
        for i in range(self.num_layers):
            attn_output = self.mha_layers[i](query=x, key=enc_output, value=enc_output, training=training)
            attn_output = self.dropout_layers[i](attn_output, training=training)
            out1 = self.layernorm1_layers[i](x + attn_output)
            ffn_output = self.ffn_layers[i](out1, training=training)
            ffn_output = self.dropout_layers[i](ffn_output, training=training)
            x = self.layernorm2_layers[i](out1 + ffn_output)
        return x

def create_model(vocab_size, max_length):
    embedding_dim = 256

    # image feature: (2048,)
    inputs1 = tf.keras.Input(shape=(2048,), name="img_feat")
    fe1 = Dense(256, activation='relu')(inputs1)
    fe2 = RepeatVector(max_length - 1, name="repeat_vector")(fe1)  # (None, maxlen-1, 256)

    # caption tokens
    inputs2 = tf.keras.Input(shape=(max_length - 1,), name="caption_in")
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)

    decoder = TransformerDecoder(num_layers=3, d_model=embedding_dim, num_heads=4, dff=512, rate=0.2, name="transformer_decoder")
    dec_output = decoder([se2, fe2])  # IMPORTANT: pass as list

    outputs = Dense(vocab_size, activation='softmax')(dec_output)
    return tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs, name="img_caption_transformer")
