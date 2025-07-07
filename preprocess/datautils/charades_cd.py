import json
import nltk
from collections import Counter
import os
import pickle
import numpy as np
import torch
from openie import StanfordOpenIE
from keybert import KeyBERT
from preprocess.datautils.clip_encoder import CLIP_encode_text
from preprocess.datautils.glove_encoder import GloVe_encode_text, post_process_text
from tqdm import tqdm


def process_query_oie_clip(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as f:
        lines = f.readlines()
    instances = []
    for i in tqdm(range(len(lines)), desc=f"Load Charades-CD {args.mode} annotations"):
        meta = lines[i].split("##")
        sentence = meta[1].rstrip()
        instances.append(sentence)

    # Either create the vocab or load it from disk
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')
            query_oie_subject_token_to_id = {}
            query_oie_relation_token_to_id = {}
            query_oie_object_token_to_id = {}
            for instance in tqdm(instances):
                query = instance.lower()[:-1]
                if client.annotate(query) == []:
                    q = query.split()
                    token_subject = ' '.join(q[0:round(len(q) / 3)])
                    token_relation = ' '.join(q[round(len(q) / 3):2 * round(len(q) / 3)])
                    token_object = ' '.join(q[2 * round(len(q) / 3):len(q)])
                else:
                    token_subject = client.annotate(query)[0]['subject']
                    token_relation = client.annotate(query)[0]['relation']
                    token_object = client.annotate(query)[0]['object']

                if token_subject == []:
                    token_subject = query
                if token_relation == []:
                    token_relation = query
                if token_object == []:
                    token_object = query

                # subject
                sub_words_id, sub_words_weight = \
                    args.tokenizer.tokenize(token_subject, max_valid_length=args.max_sub_words_l)
                token_subject_words = nltk.word_tokenize(token_subject)
                for i in range(len(token_subject_words)):
                    token = token_subject_words[i]
                    if token not in query_oie_subject_token_to_id:
                        query_oie_subject_token_to_id[token] = int(sub_words_id[0][i])

                # relation
                rel_words_id, rel_words_weight = \
                    args.tokenizer.tokenize(token_relation, max_valid_length=args.max_rel_words_l)
                token_relation_words = nltk.word_tokenize(token_relation)
                for i in range(len(token_relation_words)):
                    token = token_relation_words[i]
                    if token not in query_oie_relation_token_to_id:
                        query_oie_relation_token_to_id[token] = int(rel_words_id[0][i])

                # object
                obj_words_id, obj_words_weight = \
                    args.tokenizer.tokenize(token_object, max_valid_length=args.max_obj_words_l)
                token_object_words = nltk.word_tokenize(token_object)
                for i in range(len(token_object_words)):
                    token = token_object_words[i]
                    if token not in query_oie_object_token_to_id:
                        query_oie_object_token_to_id[token] = int(obj_words_id[0][i])

            print('Get query_oie_subject_token_to_id')
            print(len(query_oie_subject_token_to_id))
            print('Get query_oie_relation_token_to_id')
            print(len(query_oie_relation_token_to_id))
            print('Get query_oie_object_token_to_id')
            print(len(query_oie_object_token_to_id))

            print('Write into %s' % args.vocab_subject_json.format(args.dataset, args.dataset))
            with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'w') as f:
                json.dump(query_oie_subject_token_to_id, f, indent=4)

            print('Write into %s' % args.vocab_relation_json.format(args.dataset, args.dataset))
            with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'w') as f:
                json.dump(query_oie_relation_token_to_id, f, indent=4)

            print('Write into %s' % args.vocab_object_json.format(args.dataset, args.dataset))
            with open(args.vocab_object_json.format(args.dataset, args.dataset), 'w') as f:
                json.dump(query_oie_object_token_to_id, f, indent=4)

        else:
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


def process_query_oie_glove(args):
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

        sub_words = list(query_oie_subject_token_to_id.keys())
        sub_words_feat = args.tokenizer.tokenize(sub_words)
        obj = {
            'feat': sub_words_feat.numpy(),
            'words_id': list(query_oie_subject_token_to_id.values())
        }
        print('Writing', args.output_subject_pkl.format(args.dataset, args.dataset, args.mode))
        with open(args.output_subject_pkl.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

        rel_words = list(query_oie_relation_token_to_id.keys())
        rel_words_feat = args.tokenizer.tokenize(rel_words)
        obj = {
            'feat': rel_words_feat.numpy(),
            'words_id': list(query_oie_relation_token_to_id.values())
        }
        print('Writing', args.output_relation_pkl.format(args.dataset, args.dataset, args.mode))
        with open(args.output_relation_pkl.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

        obj_words = list(query_oie_object_token_to_id.keys())
        obj_words_feat = args.tokenizer.tokenize(obj_words)
        obj = {
            'feat': obj_words_feat.numpy(),
            'words_id': list(query_oie_object_token_to_id.values())
        }
        print('Writing', args.output_object_pkl.format(args.dataset, args.dataset, args.mode))
        with open(args.output_object_pkl.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)



