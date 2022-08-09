
import os

path_ = os.path.dirname(os.path.realpath(__file__))

data_base_dir = os.path.join(path_, '../data')
dataset_base_dir = os.path.join(data_base_dir, "dataset")

fm_qa_dir = os.path.join(dataset_base_dir, "fm_qa")
fm_qa_json_path = os.path.join(fm_qa_dir, "FM-CH-QA.json")
fm_qa_vocab_path = os.path.join(fm_qa_dir, "vocab.txt")

fm_qa_train_src_path = os.path.join(fm_qa_dir, "train_src_input.pkl")
fm_qa_train_trg_input_path = os.path.join(fm_qa_dir, "train_trg_input.pkl")
fm_qa_train_trg_label_path = os.path.join(fm_qa_dir, "train_trg_label.pkl")

fm_qa_val_src_path = os.path.join(fm_qa_dir, "val_src_input.pkl")
fm_qa_val_trg_input_path = os.path.join(fm_qa_dir, "val_trg_input.pkl")
fm_qa_val_trg_label_path = os.path.join(fm_qa_dir, "val_trg_label.pkl")

