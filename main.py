import tensorflow as tf
import utils
import models
import numpy as np

train_src_ids = utils.fm_qa_data_process.load_pickle(utils.path.fm_qa_train_src_path)
train_trg_input_ids = utils.fm_qa_data_process.load_pickle(utils.path.fm_qa_train_trg_input_path)
train_trg_label_ids = utils.fm_qa_data_process.load_pickle(utils.path.fm_qa_train_trg_label_path)

train_src_ids = np.array(train_src_ids)
train_trg_input_ids = np.array(train_trg_input_ids)
train_trg_label_ids = np.array(train_trg_label_ids)

seq_model = models.seq2seq_v101.Seq2Seq(vocab_size=utils.fm_qa_data_process.vocab_size)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

seq_model.compile(optimizer=opt, loss=loss, metrics=['acc'])

for i in range(1, 11):
    seq_model.fit(x=[train_src_ids, train_trg_input_ids],
                  y=train_trg_label_ids,
                  batch_size=32, epochs=10)
    seq_model.save_weights(f"data/model/seq_v101_{10*i}_wights.h5")




