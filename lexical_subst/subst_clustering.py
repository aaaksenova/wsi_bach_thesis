from generate_subst import generate
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import random
import torch


def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

set_all_seeds(42)


def max_ari(df, X, ncs,
            affinity='cosine', linkage='average', vectorizer=None):
    """
    Gets data and substitutions (and some parameters
    like vectorizer and parameters for clusterization).
    Here for each unique word substitutions
    are vectorized and senses are clusterized.
    It returns metrics of clusterization.
    """
    sdfs = []
    for word in df.word.unique():
        # collecting examples for the word
        mask = (df.word == word)
        #         print(mask)
        # vectors for substitutions
        vectors = X[mask] if vectorizer is None \
            else vectorizer.fit_transform(X[mask]).toarray()
        # vectors_ling = np.hstack((vectors, df[mask][['case_enc', 'number_enc', 'dep_enc']].to_numpy()))
        # ids of senses of the examples
        gold_sense_ids = df.gold_sense_id[mask]
        gold_sense_ids = None if gold_sense_ids.isnull().any() \
            else gold_sense_ids

        # clusterization (ari is kept in sdf along with other info)
        best_clids, sdf, _ = clusterize_search(word, vectors, gold_sense_ids,
                                               ncs=ncs,
                                               affinity=affinity, linkage=linkage)
        df.loc[mask, 'predict_sense_id'] = best_clids  # result ids of clusters
        sdfs.append(sdf)

    return sdfs


def clusterize_search(word, vecs, gold_sense_ids=None,
                      ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
                      affinity='cosine', linkage='average', print_topf=None,
                      generate_pictures_df=False, corpora_ids=None):
    """
    Gets word, vectors, gold_sense_ids and provides AgglomerativeClustering.
    """

    sdfs = []

    # adding 1 to zero vectors, because there will be problems (all-zero vectorized entries), if they remain zero
    # this introduces a new dimension with possible values
    # 1 -- all the other coords are zeros
    # 0 -- other coords are not all-zeros
    zero_vecs = ((vecs ** 2).sum(axis=-1) == 0)
    if zero_vecs.sum() > 0:
        vecs = np.concatenate((vecs,
                               zero_vecs[:, np.newaxis].astype(vecs.dtype)),
                              axis=-1)

    best_clids = None
    best_silhouette = 0
    distances = []

    # matrix with computed distances between each pair of the two collections of inputs
    distance_matrix = cdist(vecs, vecs, metric=affinity)
    distances.append(distance_matrix)

    for nc in ncs:
        # clusterization
        clr = AgglomerativeClustering(affinity='precomputed',
                                      linkage=linkage,
                                      n_clusters=nc)
        clids = clr.fit_predict(distance_matrix) if nc > 1 \
            else np.zeros(len(vecs))

        # computing metrics
        ari = ARI(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
        sil_cosine = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='cosine')
        sil_euclidean = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='euclidean')

        # vc like 5/4/3 says that
        # there are 5 examples w golden_id=id1;
        # there are 4 examples w golden_id=id2;
        # there are 3 examples w golden_id=id3;
        # e.g. вид 4/3/2 means that there were
        # 4 examples with вид==view; 3 examples .==type; 2 examples .==specie
        vc = '' if gold_sense_ids is None else '/'.join(
            np.sort(pd.value_counts(gold_sense_ids).values)[::-1].astype(str))

        if sil_cosine > best_silhouette:
            best_silhouette = sil_cosine
            best_clids = clids

        # metrics for each word
        sdf = pd.DataFrame({'ari': ari,
                            'word': word, 'nc': nc,
                            'sil_cosine': sil_cosine,
                            'sil_euclidean': sil_euclidean,
                            'vc': vc,
                            'affinity': affinity, 'linkage': linkage},
                           index=[0])

        sdfs.append(sdf)

    sdf = pd.concat(sdfs, ignore_index=True)

    return best_clids, sdf, distances


def metrics(sdfs):
    """
    Gets dataset and calculates ari and silhouette score for all lexemes.
    """
    # all metrics for each unique word
    sdf = pd.concat(sdfs, ignore_index=True)
    # groupby is docuented to preserve inside group order
    res = sdf.sort_values(by='ari').groupby(by='word').last()
    # maxari for fixed hypers
    fixed_hypers = sdf.groupby(['affinity',
                                'linkage',
                                'nc']).agg({'ari': np.mean}).reset_index()
    idxmax = fixed_hypers.ari.idxmax()
    res_df = fixed_hypers.loc[idxmax:idxmax].copy()
    res_df = res_df.rename(columns=lambda c: 'fh_maxari' if c == 'ari' \
                           else 'fh_' + c)
    res_df['maxari'] = res.ari.mean()

    for metric in [c for c in sdf.columns if c.startswith('sil')]:
        res_df[metric+'_ari'] = sdf.sort_values(by=metric).groupby(by='word').last().ari.mean()

    return res_df, res, sdf


def run_pipeline(path, model, top_k):
    """
    Combines generation, preprocessing and clustering in one pipeline.
    """
    df = generate(path, model, top_k) #('/Users/a19336136/PycharmProjects/ling_wsi/wsi_bach_thesis/russe-wsi-kit/data/main/bts-rnc/train.csv',
                  #'cointegrated/rubert-tiny')
    print('Substitutions are generated')
    subst_texts = df['subst_texts']
    print('Substitutions are processed')
    vectorizer = 'TfidfVectorizer'
    lemmatize = True
    analyzer = 'word'
    min_df = 0.05
    max_df = 0.95
    ngram_range = (1, 1)
    ncs = (2, 10)

    vec = eval(vectorizer)(token_pattern=r"(?u)\b\w+\b",
                           min_df=min_df, max_df=max_df,
                           analyzer=analyzer, ngram_range=ngram_range)

    sdfs = max_ari(df,
                   subst_texts,
                   ncs=range(*ncs),
                   affinity='cosine',
                   linkage='average',
                   vectorizer=vec)

    res_df, res, sdf = metrics(sdfs)
    res_df.to_csv('result/res_overall.csv', sep='\t')
    res.to_csv('result/res_detailed.csv', sep='\t')
    print('Clustering finished')
