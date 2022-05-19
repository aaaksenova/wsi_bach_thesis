import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import random
import torch
from generate_subst import generate
import os
import umap
import plotly.express as px
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_all_seeds(42)


def max_ari(df, X, ncs,
            affinity='cosine', linkage='average', methods=None, vectorizer=None,
            ncs_search=True):
    """
    Gets data and set of profiling methods.
    Grammatical profiles are vectorized
    and senses are clustered.
    It returns metrics of clustering.
    """
    df['subst_vecs'] = [[0]] * df.shape[0]
    sdfs = []
    methods = methods.split('_')
    if 'subst' in methods:
        flag_subst = methods.pop(methods.index('subst'))
    else:
        flag_subst = None
    if 'ling' in methods:
        flag_ling = methods.pop(methods.index('ling'))
    else:
        flag_ling = None
    if 'prep' in methods:
        flag_prep = methods.pop(methods.index('prep'))
    else:
        flag_prep = None
    if 'headvec' in methods:
        flag_headvec = methods.pop(methods.index('headvec'))
    else:
        flag_headvec = None
    if 'headling' in methods:
        flag_headling = methods.pop(methods.index('headling'))
    else:
        flag_headling = None
    morph_start = 'Anim'
    morph_end = 'Sing'
    synt_start = "acl"
    synt_end = "xcomp"
    child_start = "acl_child"
    child_end = "xcomp_child"
    m = {'morph': [morph_start, morph_end],
         'synt': [synt_start, synt_end],
         'child': [child_start, child_end]
         }

    # if not flag_subst:
    try:
        df_profiles = df.loc[:, m[methods[0]][0]: m[methods[0]][1]].astype('float')
    except Exception as e:
        print(e)
    try:
        cols = [i for i in df.columns.tolist() if 'target_' in i]
        df_ling_target = df.loc[:, cols[0]: cols[-1]].astype('int64')
    except Exception as e:
        print(e)

        if len(methods) > 1:
            for method in methods[1:]:
                df_profiles.join(df.loc[:, m[method][0]: m[method][1]].astype('float'))
    for word in df.word.unique():
        vectors_ling = np.array([[]])
        vectors_ling_target = np.array([[]])
        vectors_prep = np.array([[]])
        vectors = np.array([[]])
        vectors_headling = np.array([[]])
        vectors_headvec = np.array([[]])
        # collecting examples for the word
        mask = (df.word == word)
        if len(methods) >= 1:
            vectors_ling = df_profiles[mask].to_numpy()
        if flag_subst:
            vectors = X[mask] if vectorizer is None \
                else vectorizer.fit_transform(X[mask]).toarray()
            df.loc[mask, 'subst_vecs'] = [frozenset(i) for i in vectors]
        if flag_ling:
            vectors_ling_target = df_ling_target[mask].to_numpy()
        if flag_prep:
            vectors_prep = np.stack(df.prep_vec[mask].tolist())
        if flag_headvec:
            vectors_headvec = np.stack(df.head_vec[mask].tolist())
        if flag_headling:
            vectors_headling = df[['head_pos', 'head_deprel']][mask].to_numpy()
        stack_list = [i for i in [vectors, vectors_ling, vectors_ling_target,
                                  vectors_prep, vectors_headvec, vectors_headling] if len(i[0]) > 0]

        vector_for_clustering = np.hstack(stack_list)

        # ids of senses of the examples
        gold_sense_ids = df.gold_sense_id[mask]
        gold_sense_ids = None if gold_sense_ids.isnull().any() \
            else gold_sense_ids

        if not ncs_search:
            ncs = [df.num_senses[mask].to_list()[0]]
        # clusterization (ari is kept in sdf along with other info)
        best_clids, sdf, _ = clusterize_search(word, vector_for_clustering, gold_sense_ids,
                                               ncs=ncs,
                                               affinity=affinity, linkage=linkage, ncs_search=ncs_search)

        df.loc[mask, 'predict_sense_id'] = best_clids
        # print(word, len(set(best_clids)))
        sdfs.append(sdf)
    df['predict_sense_id'].fillna(0, inplace=True)
    df['predict_sense_id'] = df['predict_sense_id'].astype('int64')

    return sdfs, df


def clusterize_search(word, vecs, gold_sense_ids=None,
                      ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
                      affinity='cosine', linkage='average', ncs_search=True):
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
        if nc >= len(vecs):
            print(f"We have only {len(vecs)} samples")
            break
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
    res = sdf.sort_values(by='sil_cosine').groupby(by='word').last()
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
        res_df[metric + '_ari'] = sdf.sort_values(by=metric).groupby(by='word').last().ari.mean()

    return res_df, res, sdf


def format_str(df):
    """
    Gets dataset and creates string formatting for html visualization
    """
    string = df['context']
    target_word = df['word']
    ret = []
    i = 0
    processed_string = []
    for word in string.split():
        if word == target_word:
            processed_string.append('<b>')
            processed_string.append(word)
            processed_string.append('</b>')
        else:
            processed_string.append(word)
    for word in processed_string:
        if i == 12:
            ret.append('<br>')
            i = 0
        i += 1
        ret.append(word)
    return " ".join(ret)


def get_feature_mapping(df, methods):
    """
    Maps method keywords to dataframe columns
    returns list of float type columns and list of array-like columns
    """
    dict_mapping = {'subst' : ['subst_vecs'],
                    'ling': [i for i in df.columns.tolist() if 'target_' in i],
                    'morph': ['Anim', 'Inan', 'Acc', 'Dat', 'Gen', 'Ins', 'Loc', 'Nom', 'Par', 'Voc',
            'Fem', 'Masc', 'Neut', 'Plur', 'Sing'],
                    'child': [
            "acl_child",
            "advcl_child",
            "advmod_child",
            "amod_child",
            "appos_child",
            "aux_child",
            "case_child",
            "cc_child",
            "ccomp_child",
            "clf_child",
            "compound_child",
            "conj_child",
            "cop_child",
            "csubj_child",
            "dep_child",
            "det_child",
            "discourse_child",
            "dislocated_child",
            "expl_child",
            "fixed_child",
            "flat_child",
            "goeswith_child",
            "iobj_child",
            "list_child",
            "mark_child",
            "nmod_child",
            "nsubj_child",
            "nummod_child",
            "obj_child",
            "obl_child",
            "orphan_child",
            "parataxis_child",
            "punct_child",
            "reparandum_child",
            "root_child",
            "vocative_child",
            "xcomp_child"],
                    'synt': ["acl",
            "advcl",
            "advmod",
            "amod",
            "appos",
            "aux",
            "case",
            "cc",
            "ccomp",
            "clf",
            "compound",
            "conj",
            "cop",
            "csubj",
            "dep",
            "det",
            "discourse",
            "dislocated",
            "expl",
            "fixed",
            "flat",
            "goeswith",
            "iobj",
            "list",
            "mark",
            "nmod",
            "nsubj",
            "nummod",
            "obj",
            "obl",
            "orphan",
            "parataxis",
            "punct",
            "reparandum",
            "root",
            "vocative",
            "xcomp"],
                    'headling': ['head_pos', 'head_deprel'],
                    'headvec': ['head_vec'],
                    'prep': ['prep_vec']}
    features = []
    for m in methods.split('_'):
        if m not in ['subst', 'headvec', 'prep']:
            features.extend(dict_mapping[m])
    features_vec = []
    for m in methods.split('_'):
        if m in ['subst', 'headvec', 'prep']:
            features_vec.extend(dict_mapping[m])
    return features, features_vec


def make_html_picture(df, path, methods):
    """
    Creates html visualization for each word in one clustering experiment.
    """
    features, features_vec = get_feature_mapping(df, methods)
    df['format_context'] = df.apply(format_str, axis=1)
    df['gold_sense_id'] = df['gold_sense_id'].astype(str)
    df['predict_sense_id'] = df['predict_sense_id'].astype(str)

    for word in df.word.unique():
        mask = (df.word == word)
        df1 = df[mask].copy()
        feats_for_clustering = df1[features].to_numpy()
        if len(features_vec) > 0:
            for i in features_vec:

                if i == 'subst_vecs':
                    maxlen = max([len(j) for j in df1[i].tolist()])
                    feat_vec = np.vstack([list(j) + [0]*(maxlen-len(j)) for j in df1[i].tolist()])

                else:
                    feat_vec = df1[i].to_numpy()
                feats_for_clustering = np.hstack((feats_for_clustering, feat_vec))

        reducer = umap.UMAP(metric='cosine', random_state=42)
        mapped = reducer.fit_transform(feats_for_clustering)
        # tsne = TSNE(n_components=2, init='pca').fit_transform(feats_for_clustering)

        df1['umap_x'] = [el[0] for el in mapped]
        df1['umap_y'] = [el[1] for el in mapped]

        fig = px.scatter(df1,
                         x="umap_x",
                         y="umap_y",
                         color="gold_sense_id",
                         facet_col='word',
                         symbol='predict_sense_id',
                         text='format_context')

        fig.update_traces(mode="markers", hovertemplate=None)
        fig.write_html(f'{path}/{word}_visualization.html')


def run_pipeline(path, modelname, top_k, methods, detailed_analysis=False):
    """
    Path: path to dataset
    Model: higgingface model
    Top-k: number of substitutes to generate
    Methods: list of methods separeted by '_'
    e.g. morph_synt_child_subst
    """

    df = generate(path, modelname, top_k, methods)
    df_senses = pd.read_csv(path, sep='\t')
    print('Data processing finished')
    subst_texts = df['subst_texts']
    if 'num_senses' in df_senses.columns:
        df['num_senses'] = df_senses['num_senses']
        print('Num senses fixed')
        ncs = 0
        ncs_search = False
    else:
        ncs = (2, 10)
        ncs = range(*ncs)
        ncs_search = True

    vectorizer = 'TfidfVectorizer'
    lemmatize = True
    analyzer = 'word'
    min_df = 0.05
    max_df = 0.95
    ngram_range = (1, 1)

    vec = eval(vectorizer)(token_pattern=r"(?u)\b\w+\b",
                           min_df=min_df, max_df=max_df,
                           analyzer=analyzer, ngram_range=ngram_range)
    # ncs = (2, 10)
    sdfs, df_predicted = max_ari(df,
                                 subst_texts,
                                 ncs=ncs,
                                 affinity='cosine',
                                 linkage='average',
                                 methods=methods,
                                 vectorizer=vec,
                                 ncs_search=ncs_search)

    res_df, res, sdf = metrics(sdfs)

    if detailed_analysis:
        if not os.path.exists(f'detailed_clustering_analysis/{modelname.split("/")[-1]}_{methods}'):
            os.mkdir(f'detailed_clustering_analysis/{modelname.split("/")[-1]}_{methods}')
        res_df.to_csv(
            f'detailed_clustering_analysis/{modelname.split("/")[-1]}_{methods}/res_overall_{modelname.split("/")[-1]}_{methods}.tsv',
            sep='\t')
        res.to_csv(
            f'detailed_clustering_analysis/{modelname.split("/")[-1]}_{methods}/res_detailed_{modelname.split("/")[-1]}_{methods}.tsv',
            sep='\t')
        df_predicted[['word', 'context', 'positions', 'gold_sense_id', 'predict_sense_id']].to_csv(
            f'detailed_clustering_analysis/{modelname.split("/")[-1]}_{methods}/predicted_{modelname.split("/")[-1]}_{methods}.tsv',
            sep='\t', index=False)
        make_html_picture(df_predicted, f'detailed_clustering_analysis/{modelname.split("/")[-1]}_{methods}/',
                          methods=methods)

    else:
        if not os.path.exists('result'):
            os.mkdir('result')
        res_df.to_csv(f'result/res_overall_{modelname.split("/")[-1]}_{methods}.tsv', sep='\t')
        res.to_csv(
            f'result/res_detailed_{modelname.split("/")[-1]}_{methods}.tsv',
            sep='\t')
    print('Clustering finished')
