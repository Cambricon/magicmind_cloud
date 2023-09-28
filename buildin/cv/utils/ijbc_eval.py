import numpy as np
import os
import timeit
from sklearn.metrics import roc_curve, auc
import pandas as pd
from tqdm import tqdm
from utils import Record

IJB_DATASETS_PATH = os.environ.get("IJB_DATASETS_PATH")

def load_feature(features_dir, img_id):
    feature_file = features_dir + '/' + str(img_id) + '.feature'
    feature = np.fromfile(feature_file, dtype=float, count=-1, sep=' ')
    # faceness_score = feature[512]
    feature = feature[0:512] * feature[512]
    return feature

def load_features(features_dir, img_ids):
    from multiprocessing import Pool, cpu_count
    from functools import partial
    import math
    process_num = int(math.ceil(cpu_count() * 0.9))
    func = partial(load_feature, features_dir)
    with Pool(process_num) as p:
        features = list(tqdm(p.imap(func, img_ids), total=len(img_ids), desc='load features'))
    features = np.array(features, dtype=float)
    print('features shape:', features.shape)
    return features

def summary(features_dir, template_media_list_path,
            read_template_pair_list_path):
    templates, medias = read_template_media_list(template_media_list_path)
    p1, p2, label = read_template_pair_list(read_template_pair_list_path)
    
    with open(template_media_list_path, "r") as f:
        lines = f.readlines()
        img_ids = [int(id.split(" ")[0].split(".")[0]) for id in lines]
     
    features = load_features(features_dir, img_ids)

    template_norm_feats, unique_templates = image2template_feature(features, templates, medias)

    score = verification(template_norm_feats, unique_templates, p1, p2)
    methods = np.array(["ijbc"])
    scores = dict(zip(methods, [score]))

    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    y_labels = []

    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(
                list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            y_labels.append('%.5f' % (tpr[min_index] * 100))
    return {"1e-5": y_labels[1], "1e-4": y_labels[2]}, y_labels[1], y_labels[2]


def read_template_pair_list(path):
    import pandas as pd
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_template_media_list(path):
    import pandas as pd
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def image2template_feature(img_feats=None, templates=None, medias=None):
    from sklearn import preprocessing
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t, ) = np.where(templates == uqt)
        # max_value = np.max(ind_t)
        #(ind_t, ) = np.where(ind_t < img_feats.shape[0])
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def main():
    import argparse
    args = argparse.ArgumentParser(description='evaluate IJBC')
    args.add_argument('--features_dir', default = 'output_features',
            required = True, type = str, help = 'features directory')
    args.add_argument('--output_file', default = 'output_file',
            required = True, type = str, help = 'result file')
    args.add_argument('--face_tid_mid_file', default = str(IJB_DATASETS_PATH) + '/IJBC/meta/ijbc_face_tid_mid.txt',
            required = False, type = str, help = 'result file')
    args.add_argument('--template_pair_label_file', default = str(IJB_DATASETS_PATH) + '/IJBC/meta/ijbc_template_pair_label.txt',
            required = False, type = str, help = 'result file')
    args = args.parse_args()
    res, res_1e5, res_1e4 = summary(args.features_dir,  args.face_tid_mid_file, args.template_pair_label_file)
    print(res)
    result_file = Record(args.output_file)
    result_file.write("IJB-C(1E-5): %s"%(res_1e5), False)
    result_file.write("IJB-C(1E-4): %s"%(res_1e4), False)

if __name__ == "__main__":
    main()

