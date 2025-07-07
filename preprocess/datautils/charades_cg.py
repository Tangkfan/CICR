import json
from preprocess.datautils import clip_encoder
import nltk
from collections import Counter
import os
import pickle
import numpy as np
import jsonlines
import pandas as pd
import torch
from openie import StanfordOpenIE
from preprocess.datautils.clip_encoder import CLIP_encode_text
from tqdm import tqdm


def process_query_oie(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as f:
        json_obj = json.load(f)

    instances = []
    for video_id in tqdm(json_obj.keys(), desc=f"Load TACoS {args.mode} annotations"):
        meta = json_obj[video_id]
        for sentence in meta['sentences']:
            instances.append(sentence)

    # Either create the vocab or load it from disk
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:

        print('Loading oie_subject vocab')
        with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'r') as f:
            query_oie_subject_token_to_id = json.load(f)
        print('Loading oie_relation vocab')
        with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'r') as f:
            query_oie_relation_token_to_id = json.load(f)
        print('Loading oie_object vocab')
        with open(args.vocab_object_json.format(args.dataset, args.dataset), 'r') as f:
            query_oie_object_token_to_id = json.load(f)
        # Encode all queries
        print('Encoding data')

        sub_words_id = list(query_oie_subject_token_to_id.values())
        sub_words_id = torch.tensor(sub_words_id, dtype=torch.int).unsqueeze(0).cuda()
        sub_words_feat, sub_words_id = \
            CLIP_encode_text(args, sub_words_id, device=sub_words_id.device)
        obj = {
            'feat': sub_words_feat,
            'words_id': sub_words_id
        }
        print('Writing', args.output_subject_pkl.format(args.dataset, args.dataset, args.mode))
        with open(args.output_subject_pkl.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

        rel_words_id = list(query_oie_relation_token_to_id.values())
        rel_words_id = torch.tensor(rel_words_id, dtype=torch.int).unsqueeze(0).cuda()
        rel_words_feat, rel_words_id = \
            CLIP_encode_text(args, rel_words_id, device=rel_words_id.device)
        obj = {
            'feat': rel_words_feat,
            'words_id': rel_words_id
        }
        print('Writing', args.output_relation_pkl.format(args.dataset, args.dataset, args.mode))
        with open(args.output_relation_pkl.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

        obj_words_id = list(query_oie_object_token_to_id.values())
        obj_words_id = torch.tensor(obj_words_id, dtype=torch.int).unsqueeze(0).cuda()
        obj_words_feat, obj_words_id = \
            CLIP_encode_text(args, obj_words_id, device=obj_words_id.device)
        obj = {
            'feat': obj_words_feat,
            'words_id': obj_words_id
        }
        print('Writing', args.output_object_pkl.format(args.dataset, args.dataset, args.mode))
        with open(args.output_object_pkl.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

