import tensorflow as tf


class Seq2Seq(tf.keras.models.Model):
    """ 简单的Seq2Seq模型
    Encoder: Embedding + LSTM
    Decoder: Embedding + LSTM + Dense
    """

    def __init__(self, vocab_size, embed_dim=128, lstm_units=256):
        super().__init__()

        # 定义 Encoder、Decoder EMbedding
        self.encoder_embed = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                       output_dim=embed_dim)
        self.decoder_embed = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                       output_dim=embed_dim)

        # 定义Encoder、Decoder LSTM
        self.encoder_lstm = tf.keras.layers.LSTM(units=lstm_units, return_state=True)
        self.bidirect_layer = tf.keras.layers.Bidirectional(self.encoder_lstm, merge_mode='sum')

        self.decoder_lstm = tf.keras.layers.LSTM(units=lstm_units, return_state=True)

        # 定义全连接层
        self.dense = tf.keras.layers.Dense(units=vocab_size, activation='softmax')

    def call(self, inputs, start_index=2, training=True):
        """ 计算，运算
        inputs: 输入数据，
            train: [src, trg]
            predict: src
        """
        if training:
            src_b, trg_b = inputs
        else:
            src_b = inputs
        e_e = self.encoder_embed(src_b)
        _, f_hid, f_cell, b_hid, b_cell = self.bidirect_layer(e_e)
        hidden = tf.add(f_hid, b_hid)
        cell = tf.add(f_cell, b_cell)
        t_outputs = []
        for t in range(src_b.shape[1]):
            if training or t == 0:
                # t== 0 这是一个隐患，当 src 开始 和 trget input 开始 index 不一致的时候，会有问题
                x = tf.slice(src_b, [0, t], [-1, 1])  # 每次取一个steps
            d_e = self.decoder_embed(x)
            out_put, hidden, cell = self.decoder_lstm(d_e, initial_state=[hidden, cell])
            out = self.dense(out_put)
            if not training:
                x = tf.argmax(out, axis=-1)
                x = tf.expand_dims(x, -1)
            out = tf.expand_dims(out, -2)  # [batch, vocab] -> [batch, 1, vocab]
            t_outputs.append(out)
        f_out = tf.concat(t_outputs, axis=1)
        return f_out



""" 训练记录
数据集说明：1万条，排序的短文本数据 

- src input: <GO> texts <EOS>
- trg input: <GO> texts 
- label: texts <EOS>

from data_process.sort_process import SortProcess
from data_process.data_iter import DataIter 
import numpy as np

src_path = "data/sort/letters_source.txt"
trg_path = "data/sort/letters_target.txt"

sp = SortProcess(src_path, trg_path)
src_ids, trg_ids, label_ids = sp.dataset(max_len=10, to_numpy=False)
src_ids = np.array(src_ids)
trg_ids = np.array(trg_ids)
label_ids = np.array(label_ids)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
model = Seq2Seq(vocab_size=len(sp.i2c))
model.compile(optimizer=opt, loss=loss, metrics=['acc'])
model.fit(x=[src_ids, trg_ids], y=label_ids, batch_size=32, epochs=10)

Epoch 1/10
313/313 [==============================] - 16s 30ms/step - loss: 1.1488 - acc: 0.6569
Epoch 2/10
313/313 [==============================] - 10s 31ms/step - loss: 0.2789 - acc: 0.9222
Epoch 3/10
313/313 [==============================] - 10s 32ms/step - loss: 0.0619 - acc: 0.9901
Epoch 4/10
313/313 [==============================] - 10s 31ms/step - loss: 0.0244 - acc: 0.9978
Epoch 5/10
313/313 [==============================] - 10s 31ms/step - loss: 0.0091 - acc: 0.9998
Epoch 6/10
313/313 [==============================] - 10s 31ms/step - loss: 0.0051 - acc: 1.0000
Epoch 7/10
313/313 [==============================] - 10s 31ms/step - loss: 0.0505 - acc: 0.9872
Epoch 8/10
313/313 [==============================] - 10s 31ms/step - loss: 0.0034 - acc: 1.0000
Epoch 9/10
313/313 [==============================] - 10s 32ms/step - loss: 0.0020 - acc: 1.0000
Epoch 10/10
313/313 [==============================] - 10s 31ms/step - loss: 0.0015 - acc: 1.0000
"""




