import utils
import os
import json

pad_flag = '[PAD]'
unk_flag = '[UNK]'
begin_flag = '[GO]'
end_flag = '[EOS]'

import pickle


def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data


def read_json_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    return json_data


def read_text_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf8') as fp:
        txt = fp.read()
    return txt


def save_text(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_q_a_from_dict(qa_dict, q_key='Question', a_key='Answer'):
    q = qa_dict.get(q_key, '')
    a = qa_dict.get(a_key, '')
    return q, a


def creat_vocab(path):
    json_data = read_json_file(path)
    char_dict = {}
    # trian
    for qa in json_data['train']:
        q, a = get_q_a_from_dict(qa)
        for c in str(a) + str(q):
            char_dict[c] = char_dict.get(c, 0) + 1

    for qa in json_data['val']:
        q, a = get_q_a_from_dict(qa)
        for c in str(a) + str(q):
            char_dict[c] = char_dict.get(c, 0) + 1

    flags = [pad_flag, unk_flag, begin_flag, end_flag]
    char_items = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
    chars = [item[0] for item in char_items]
    vocab = flags + chars
    vocab = [c.strip() for c in vocab if len(c.strip()) > 0]
    return vocab


def read_vocab(qa_path, vocab_path, ignore_exist=False):
    if not ignore_exist and os.path.exists(vocab_path):
        vocab_txt = read_text_file(vocab_path)
        vocab = vocab_txt.split("\n")
        return vocab
    else:
        vocab = creat_vocab(qa_path)
        save_text(vocab_path, "\n".join(vocab))
        return vocab


vocab = read_vocab(qa_path=utils.path.fm_qa_json_path,
                   vocab_path=utils.path.fm_qa_vocab_path,
                   ignore_exist=False)
i2c = {i: c for i, c in enumerate(vocab)}
c2i = {c: i for i, c in enumerate(vocab)}
vocab_size = len(vocab)

pad_id = c2i.get(pad_flag, 0)
unk_id = c2i.get(unk_flag, 1)
begin_id = c2i.get(begin_flag, 2)
end_id = c2i.get(end_flag, 3)


def text_to_ids(texts, max_len=32, add_begin=True, add_end=True):
    ids = []
    for line in texts:
        line = line[:max_len-2]
        id_line = [c2i.get(c, unk_id) for c in line]
        if add_begin:
            id_line = [begin_id] + id_line
        if add_end:
            id_line = id_line + [end_id]
        id_line = id_line + [pad_id] * (max_len - len(id_line))
        ids.append(id_line)
    return ids


