import os
import json
import h5py
import torch
from tqdm import tqdm
import torch.nn.functional as F

from .base import BaseDataset


"""
TACoS:
- C3D video features of MS-2D-TAN clip_len = 1, max_video_l = 1402
- Train dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 0.48
    - max_clip_len = 751.43

- Val dataset
    - CLIP text tokenizer: max_words_l = 
    - min_clip_len = 
    - max_clip_len = 

- Test dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 0.78
    - max_clip_len = 578.95
"""


class AnetCDDataset(BaseDataset):
    def __init__(self, ann_path, feat_files, split,
                 use_tef, clip_len, max_words_l, max_video_l,
                 tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                 normalize_video, contra_samples,
                 recfw, vocab_size, max_gather_size):
        super().__init__(ann_path, feat_files, split,
                         use_tef, clip_len, max_words_l, max_video_l,
                         tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                         normalize_video, contra_samples,
                         recfw, vocab_size, max_gather_size)
    
    def load_annotations(self):
        split2filename = {
            "train": "anet_train_momentdiff.jsonl",
            # "val": "charades_val.json",
            "test_iid": "anet_test_iid_momentdiff.jsonl",
            "test_ood": "anet_test_ood_momentdiff.jsonl",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])
        annotations = []
        with open(ann_file, 'r') as f:
            json_obj = [json.loads(l.strip("\n")) for l in f.readlines()]
            count = 0
            for meta in tqdm(json_obj, desc=f"Load anet-cd {self.split} annotations"):
                duration = meta["duration"]
                for timestamp, sentence in zip(meta['relevant_windows'], meta['query']):
                    if timestamp[0] > timestamp[1]:
                        continue
                    count += 1
                    words_id, words_weight, unknown_mask, words_label = \
                        self.tokenizer.tokenize(sentence, max_valid_length=self.max_words_l)
                    start_time = max(timestamp[0], 0)
                    end_time = min(timestamp[1], duration)
                    moment = [start_time, end_time]
                    if self.clip_len == -1:
                        start_idx = start_time / duration
                        end_idx = end_time / duration
                    else:
                        start_idx = int(start_time / self.clip_len)
                        end_idx = int(end_time / self.clip_len)

                    data = {
                        "video_id": meta['vid'],
                        "duration": duration,
                        "moment": moment,
                        "sentence": sentence,
                        "words_id": words_id,
                        "words_weight": words_weight,
                        "unknown_mask": unknown_mask,
                        "words_label": words_label,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "qid": None if self.split=="train" else count,
                        "relevant_windows": None if self.split=="train" else [moment],
                    }
                    annotations.append(data)
        
        return annotations

    def get_video_feat(self, video_id):
        feat_file = self.feat_files[0]
        with h5py.File(feat_file, 'r') as f:
            # feat = f[video_id][:self.max_video_l]
            feat = f[video_id][:]
            if self.normalize_video:
                feat = F.normalize(torch.from_numpy(feat).to(torch.float32), dim=1)
        return feat


