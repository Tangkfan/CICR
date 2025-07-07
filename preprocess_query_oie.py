import argparse
import numpy as np
import os
import json
import pickle

from preprocess.datautils import charades
from preprocess.datautils import charades_cg
from preprocess.datautils import charades_cd
from preprocess.datautils import tacos
from preprocess.datautils import qvhighlights
from preprocess.datautils.tokenizer import CLIPTokenizer, GloVeSimpleTokenizer, \
    NLTKTokenizer, NLTKTokenizerWithFeature, Vocabulary


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def build_vocab_from_pkl(opt):
    vocab_file = os.path.join(opt.ann_path, "glove.pkl")
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def build_vocab(opt):
    vocab_file = os.path.join(opt.ann_path, "GloVe_tokenized_count.txt")
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
        words_set = set()
        for line in lines:
            word = line.split(' ')[0]
            words_set.add(word)
    vocab = Vocabulary(words_set)
    return vocab


def load_CLIP_keep_vocab(ann_path, vocab_size):
    id2label = {}
    vocab_file = os.path.join(ann_path, "CLIP_tokenized_count.txt")
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            words_id = int(line.split(' ')[0])
            id2label[words_id] = count
            count += 1
            if count == vocab_size:
                break
    return id2label


def load_GloVe_keep_vocab(ann_path, vocab_size):
    id2label = {}
    vocab_file = os.path.join(ann_path, "GloVe_tokenized_count.txt")
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            words_id = int(line.split(' ')[1])
            id2label[words_id] = count
            count += 1
            if count == vocab_size:
                break
    id2label['<unknown>'] = vocab_size
    return id2label


def load_GloVe_pkl_keep_vocab(vocab, vocab_size):
    id2label = {}
    count = 0
    for w, _ in vocab['counter'].most_common(vocab_size):
        id2label[w] = count
        count += 1
    id2label['<unknown>'] = vocab_size
    return id2label


if __name__ == '__main__':
    # nltk.download('punkt')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='preprocess/config/charades-cd.json')
    parser.add_argument('--dataset', default='qvhighlights',
                        choices=['charades', 'charades-cg', 'charades-cd', 'qvhighlights', 'tacos'], type=str)
    parser.add_argument('--tokenizer_type', default='CLIP', type=str)
    parser.add_argument('--ann_path', default=None, type=str)
    parser.add_argument("--load_vocab_pkl", default=False, action="store_true",
                        help="Only for tokenizer_type==GloveNLTK, fasttrack")
    parser.add_argument('--bpe_path', default="./word_embeddings/bpe_simple_vocab_16e6.txt.gz", type=str)
    parser.add_argument('--text_model_path', default="./word_embeddings/clip_text_encoder.pth", type=str)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument("--max_words_l", default=32, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_sub_words_l", default=4, type=int)
    parser.add_argument("--max_rel_words_l", default=4, type=int)
    parser.add_argument("--max_obj_words_l", default=4, type=int)
    parser.add_argument("--vocab_size", type=int, default=1111)
    parser.add_argument('--output_subject_pkl', type=str, default='./data/{}/{}_query_subject_{}.pkl')
    parser.add_argument('--output_relation_pkl', type=str, default='./data/{}/{}_query_relation_{}.pkl')
    parser.add_argument('--output_object_pkl', type=str, default='./data/{}/{}_query_object_{}.pkl')
    parser.add_argument('--vocab_subject_json', type=str, default='./data/{}/{}_vocab_subject.json')
    parser.add_argument('--vocab_relation_json', type=str, default='./data/{}/{}_vocab_relation.json')
    parser.add_argument('--vocab_object_json', type=str, default='./data/{}/{}_vocab_object.json')
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'])

    args = parser.parse_args()
    if args.config_file:
        args.__dict__.update(load_json(args.config_file))
    np.random.seed(args.seed)

    if args.tokenizer_type == "GloVeSimple":
        vocab = build_vocab(args)
    elif args.tokenizer_type == "GloVeNLTK":
        if args.load_vocab_pkl:
            vocab = build_vocab_from_pkl(args)
        else:
            vocab = build_vocab(args)
    else:
        vocab = None

    if args.dataset == 'charades':
        args.annotation_file = os.path.join(args.ann_path, "charades_sta_{}.txt".format(args.mode))
        # check if data folder exists
        if not os.path.exists('./data/{}'.format(args.dataset)):
            os.makedirs('./data/{}'.format(args.dataset))
        if args.tokenizer_type == "CLIP":
            args.id2label = load_CLIP_keep_vocab(args.ann_path, args.vocab_size)
            args.tokenizer = CLIPTokenizer(args.id2label, args.bpe_path)
            charades.process_query_oie_clip(args)
        elif args.tokenizer_type == "GloVeSimple":
            args.id2label = load_GloVe_keep_vocab(args.annotation_file, args.vocab_size)
            args.tokenizer = GloVeSimpleTokenizer(args.id2label, vocab)
            charades.process_query_oie_glove(args)
        elif args.tokenizer_type == "GloVeNLTK":
            if args.load_vocab_pkl:
                args.id2label = load_GloVe_pkl_keep_vocab(vocab, args.vocab_size)
                args.tokenizer = NLTKTokenizerWithFeature(args.id2label, vocab)
                charades.process_query_oie_glove(args)
            else:
                args.id2label = load_GloVe_keep_vocab(args.ann_path, args.vocab_size)
                args.tokenizer = NLTKTokenizer(args.id2label, vocab)
                charades.process_query_oie_glove(args)
    elif args.dataset == 'Charades-CG':
        args.annotation_file = os.path.join(args.ann_path, "{}.json".format(args.mode))
        # check if data folder exists
        if not os.path.exists('./data/{}'.format(args.dataset)):
            os.makedirs('./data/{}'.format(args.dataset))
        args.id2label = load_CLIP_keep_vocab(args.ann_path, args.vocab_size)
        args.tokenizer = CLIPTokenizer(args.id2label, args.bpe_path)
        charades_cg.process_query_oie(args)
    elif args.dataset == 'Charades-CD':
        args.annotation_file = os.path.join(args.ann_path, "charades_{}.json".format(args.mode))
        # check if data folder exists
        if not os.path.exists('./data/{}'.format(args.dataset)):
            os.makedirs('./data/{}'.format(args.dataset))
        if args.load_vocab_pkl:
            args.id2label = load_GloVe_pkl_keep_vocab(vocab, args.vocab_size)
            args.tokenizer = NLTKTokenizerWithFeature(args.id2label, vocab)
            charades_cd.process_query_oie_glove(args)
        else:
            args.id2label = load_GloVe_keep_vocab(args.ann_path, args.vocab_size)
            args.tokenizer = NLTKTokenizer(args.id2label, vocab)
            charades_cd.process_query_oie_glove(args)
    elif args.dataset == 'qvhighlights':
        args.annotation_file = os.path.join(args.ann_path, "highlight_{}_release.jsonl".format(args.mode))
        # check if data folder exists
        if not os.path.exists('./data/{}'.format(args.dataset)):
            os.makedirs('./data/{}'.format(args.dataset))
        args.id2label = load_CLIP_keep_vocab(args.ann_path, args.vocab_size)
        args.tokenizer = CLIPTokenizer(args.id2label, args.bpe_path)
        qvhighlights.process_query_oie(args)
    elif args.dataset == 'tacos':
        args.annotation_file = './data/TACoS/annotations/{}.json'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('./data/{}'.format(args.dataset)):
            os.makedirs('./data/{}'.format(args.dataset))
        if args.load_vocab_pkl:
            args.id2label = load_GloVe_pkl_keep_vocab(vocab, args.vocab_size)
            args.tokenizer = NLTKTokenizerWithFeature(args.id2label, vocab)
        else:
            args.id2label = load_GloVe_keep_vocab(args.ann_path, args.vocab_size)
            args.tokenizer = NLTKTokenizer(args.id2label, vocab)
        tacos.process_query_oie(args, vocab)
