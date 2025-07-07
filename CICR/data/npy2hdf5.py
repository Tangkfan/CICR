import h5py
import glob
import numpy as np
from tqdm import tqdm


def npy2hdf5(feat_path, h5_file):
    h5_file = h5py.File(h5_file, 'w')
    
    for vid_path in tqdm(glob.glob(feat_path)):
        vid = vid_path.split('.npy')[0].split('/')[-1]
        feats = np.load(vid_path)
        h5_file.create_dataset(vid, data=feats.squeeze())
    h5_file.close()


def npz2hdf5(feat_path, h5_file):
    h5_file = h5py.File(h5_file, 'w')
    
    for vid_path in tqdm(glob.glob(feat_path)):
        vid = vid_path.split('.npz')[0].split('/')[-1]
        # feats = np.load(vid_path)['features']
        feats = np.load(vid_path)['last_hidden_state']
        h5_file.create_dataset(vid, data=feats)
    h5_file.close()

if __name__ == '__main__':
    # feat_path = '/media/hdd2/TSGV/charades/features/detr/clip_features/*'
    # h5_file = './charades/clip_image.hdf5'
    # npz2hdf5(feat_path, h5_file)

    # feat_path = '/media/hdd2/TSGV/charades/features/detr/clip_text_features/*'
    # h5_file = './charades/clip_text.hdf5'
    # npz2hdf5(feat_path, h5_file)

    # feat_path = '/media/hdd2/TSGV/charades/features/detr/slowfast_features/*'
    # h5_file = './charades/slowfast.hdf5'
    # npz2hdf5(feat_path, h5_file)

    # feat_path = '/media/hdd2/TSGV/charades/features/detr/vgg_features/rgb_features/*'
    # h5_file = './charades/vgg.hdf5'
    # npy2hdf5(feat_path, h5_file)

    # feat_path = '/media/hdd2/TSGV/QVHighlights/clip_features/*'
    # h5_file = './qvhighlights/clip_image.hdf5'
    # npz2hdf5(feat_path, h5_file)

    # feat_path = '/media/hdd2/TSGV/QVHighlights/slowfast_features/*'
    # h5_file = './qvhighlights/slowfast.hdf5'
    # npz2hdf5(feat_path, h5_file)

    # feat_path = '/media/hdd2/TSGV/QVHighlights/clip_text_features/*'
    # h5_file = './qvhighlights/clip_text.hdf5'
    # npz2hdf5(feat_path, h5_file)
    
    feat_path = '/home2/liuzh/repos/VSLNet/prepare/tacos/*'
    h5_file = './data/TACoS/c3d64_VSLNet.hdf5'
    npy2hdf5(feat_path, h5_file)

    