import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


class GloVe(object):
    """
    Attributes:
        self.glove: {str: tensor}
    """
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.dim = 300
        self.glove = self._load()
        self.glove["<PAD>"] = torch.zeros(self.dim)
        self.glove["<UNK>"] = torch.randn(self.dim)

    def get(self, word):
        if self.contains(word):
            return self.glove[word]
        else:
            return self.glove["<UNK>"]

    def contains(self, word):
        return word in self.glove.keys()

    def _load(self):
        """ Load GloVe embeddings of this vocabulary.
        """
        glove = dict()
        with open(self.glove_path, 'r') as f:
            for line in tqdm(f.readlines(), desc="Reading GloVe from {}".format(self.glove_path)):
                split_line = line.split()
                word = " ".join(split_line[0: len(split_line) - self.dim])  # some words include space
                embedding = torch.from_numpy(np.array(split_line[-self.dim:], dtype=np.float32))
                glove[word] = embedding

        return glove


class GloveTextEncoder(nn.Module):
    def __init__(self, vocab, glove):
        super(GloveTextEncoder, self).__init__()
        dim = glove.dim
        self.emb = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=dim
        )
        # freeze the GloVe embedding
        for param in self.emb.parameters():
            param.requires_grad = False

        for w in vocab.wtoi.keys():
            self.emb.weight.data[vocab.wtoi[w], :] = glove.get(w)

    def forward(self, word_ids):
        """ Get embedding from word ids, and map the embedding to out_dim.
        Args:
            word_ids: (B, L)
        Returns:
            (B, L, out_dim)
        """
        return self.emb(word_ids)


def build_GloVe_text_encoder(glove_path, vocab):
    glove = GloVe(glove_path)
    # vocab = build_vocab(opt)
    glove_text_encoder = GloveTextEncoder(vocab, glove)
    return glove_text_encoder


def GloVe_encode_text(args, words_id, vocab):
    text_encoder = build_GloVe_text_encoder(args.text_model_path, vocab)
    words_feat = text_encoder(words_id)

    words_id = words_id.squeeze().to("cpu").numpy()
    words_feat = words_feat.squeeze().to("cpu").numpy()

    return words_feat, words_id


def post_process_text(args, words_feat):
    if args.normalize_txt:
        words_feat = F.normalize(words_feat, dim=-1, p=2, eps=1e-5)

    words_feat = words_feat.squeeze().numpy()

    return words_feat
