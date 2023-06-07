#! -*- coding: utf-8 -*-
import argparse
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
import magicmind.python.runtime as mm
from mm_runner import MMRunner
from logger import Logger
logging = Logger()
maxlen = 64

def similarity(text1, text2, tokenizer, model):
    """"计算text1与text2的相似度
    """
    texts = [text1, text2]
    X, S = [], []
    for t in texts:
        x, s = tokenizer.encode(t, maxlen=maxlen)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = model([S, X])[0]
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return (Z[0] * Z[1]).sum()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--magicmind_model", type=str, default="../data/models/roformer-sim_tf_force_float32_false_2_64", help="mm model")
    args.add_argument("--vocab_file", type=str, default="../data/models/chinese_roformer-sim-char-ft_L-6_H-384_A-6/vocab.txt", help="dict path")
    args.add_argument("--device_id", type=int, default=0, help="which device use")
    args = args.parse_args()

    tokenizer = Tokenizer(args.vocab_file, do_lower_case=True)  # 建立分词器 
    model = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)  

    strings_1 = [u'今天天气不错', u'今天天气不错',u'我喜欢北京',u'我喜欢北京']
    strings_2 = [u'今天天气很好', u'今天天气不好', u'我很喜欢北京', u'我不喜欢北京']
    for i in range(len(strings_1)):
        smi = similarity(strings_1[i], strings_2[i], tokenizer, model)
        logging.info('{}, {}, similarity is : {}'.format(strings_1[i], strings_2[i], smi))

if __name__ == "__main__":
    main()