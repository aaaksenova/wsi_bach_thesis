import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import random
from pymorphy2 import MorphAnalyzer
from collections import Counter
import json
import os


morph_profiles = json.load(open('/Users/a19336136/PycharmProjects/ling_wsi/wsi_bach_thesis/gram_profiling/output/jsons/corpus_morph.json'))


def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


set_all_seeds(42)
tqdm.pandas()


def mask_before_target(idxs, line):
    """
    Gets sentence and creates masked coordination pattern before target word
    """
    start_id = int(idxs.split(',')[0].split('-')[0].strip())
    return line[:start_id] + '[MASK] а также ' + line[start_id:]


def mask_after_target(idxs, line):
    """
    Gets sentence and creates masked coordination pattern after target word
    """
    end_id = int(idxs.split(',')[0].split('-')[1].strip())
    return line[:end_id + 1] + ' а также [MASK]' + line[end_id + 1:]


def load_models(modelname):
    """
    Gets huggingface model name and uploads tokenizer and LM model
    """
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertForMaskedLM.from_pretrained(modelname)
    return tokenizer, model


def predict_masked_sent(tokenizer, model, text, top_k=5):
    """
    Gets text and returns top_k model predictions with probabilities
    """
    # Tokenize input

    text = "[CLS] %s [SEP]" % text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, 1000, sorted=True)

    out = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        if predicted_token.isalpha():
            token_weight = top_k_weights[i]
            out.append((token_weight.item(), predicted_token))
        if len(out) == top_k:
             return out

    print('less then', top_k)


# def extract_ling_feats(idx, text):
#     start_id = int(idx.split('-')[0].strip())
#     end_id = int(idx.split('-')[1].strip())
#     processed = nlp(text)
#     case = morph.parse(text[start_id:end_id+1])[0].tag.case
#     number = morph.parse(text[start_id:end_id+1])[0].tag.number
#     dep=''
#     for token in processed.iter_tokens():
#         if start_id == token.start_char:
#             dep = token.words[0].deprel
#             break
#     return case, number, dep


def intersect_sparse(substs_probs, substs_probs_y, nmasks=1, s=0, debug=False):
    """
    Combines different sets of substitutes (for different templates) using
    product of smoothed probability distributions.
    """

    vec = DictVectorizer(sparse=True)
    f1 = substs_probs.apply(lambda l: {s: p for p, s in l})
    f2 = substs_probs_y.apply(lambda l: {s: p for p, s in l})
    vec.fit(list(f1) + list(f2))
    f1, f2 = (vec.transform(list(f)) for f in (f1, f2)) # sparse matrix

    alpha1, alpha2 = ((1. - f.sum(axis=-1).reshape(-1, 1)) / 250000**nmasks \
                      for f in (f1, f2))
    prod = f1.multiply(f2) + f1.multiply(alpha2) + f2.multiply(alpha1)
    # + alpha1*alpha2 is ignored to preserve sparsity
    # finally, we don't want substs with 0
    # probs before smoothing in both distribs
    fn = np.array(vec.feature_names_)
    maxlen = (substs_probs_y.apply(len)+substs_probs.apply(len)).max()
    m = prod
    n_texts = m.shape[0]

    def reverse_argsort(mdata):
        return np.argsort(mdata)[::-1]

    idx = list()
    for text_ix in range(n_texts):
      # sparce matrices are used to preserve high performance
      # refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
      # to learn the sparse matrices indexing (i.e. what is `indptr` and `data`)
        text_sparse_indices = m.indices[m.indptr[text_ix]:m.indptr[text_ix+1]]
        text_sparse_data = m.data[m.indptr[text_ix]:m.indptr[text_ix+1]]
        text_sp_data_revsorted_ixes = reverse_argsort(text_sparse_data)
        smth = text_sparse_indices[text_sp_data_revsorted_ixes]
        idx.append(smth)

    l = list()
    for text_ix, text_sparse_ixes_sorted in enumerate(idx):
        probas = m[text_ix].toarray()[0, text_sparse_ixes_sorted]
        substs = fn[text_sparse_ixes_sorted]
        good_substs = list()
        for proba, subst in zip(probas, substs):

            if ' ' not in subst.strip():
                good_substs.append((proba, subst))
        l.append(good_substs[:10])
    return l

_ma = MorphAnalyzer()
_ma_cache = {}


def ma(s):
    """
    Gets a string with one token, deletes spaces before and
    after token and returns grammatical information about it. If it was met
    before, we would get information from the special dictionary _ma_cache;
    if it was not, information would be gotten from pymorphy2.
    """
    s = s.strip()  # get rid of spaces before and after token,
                   # pytmorphy2 doesn't work with them correctly
    if s not in _ma_cache:
        _ma_cache[s] = _ma.parse(s)
    return _ma_cache[s]


def get_nf_cnt(substs_probs):
    """
    Gets substitutes and returns normal
    forms of substitutes and count of substitutes that coresponds to
    each normal form.
    """
    nf_cnt = Counter(nf for l in substs_probs \
                     for p, s in l for nf in {h.normal_form for h in ma(s)})
    # print('\n'.join('%s: %d' % p for p in nf_cnt.most_common(10)))
    return nf_cnt


def get_normal_forms(s, nf_cnt=None):
    """
    Gets string with one token and returns set of most possible lemmas,
    all lemmas or one possible lemma.
    """
    hh = ma(s)
    if nf_cnt is not None and len(hh) > 1:  # select most common normal form
        h_weights = [nf_cnt[h.normal_form] for h in hh]
        max_weight = max(h_weights)
        return {h.normal_form for i, h in enumerate(hh) \
                if h_weights[i] == max_weight}
    else:
        return {h.normal_form for h in hh}


def preprocess_substs(r, lemmatize=True, nf_cnt=None, exclude_lemmas={}):
    """
    For preprocessing of substitutions. It gets Series of substitutions
    and probabilities, exclude lemmas from exclude_lemmas
    if it is not empty and lemmatize them if it is needed.
    """
    res = [s.strip() for p, s in r]
    if exclude_lemmas:
        res1 = [s for s in res
                if not set(get_normal_forms(s)).intersection(exclude_lemmas)]
        res = res1
    if lemmatize:
        res = [nf for s in res for nf in get_normal_forms(s, nf_cnt)]
    return res


def morph_vectors(x):
    substitutions = list(set(x['subst_texts'].split()))
    morph_dict = {'Animacy': {'Anim': 0, 'Inan': 0},
                  'Case': {'Acc': 0, 'Dat': 0, 'Gen': 0, 'Ins': 0, 'Loc': 0,
                           'Nom': 0, 'Par': 0, 'Voc': 0},
                  'Gender': {'Fem': 0, 'Masc': 0, 'Neut': 0},
                  'Number': {'Plur': 0, 'Sing': 0}}
    for word in substitutions:
        if _ma.parse(word)[0].tag.POS == 'NOUN':
            if word in morph_profiles.keys():
                for key, number in morph_profiles[word].items():
                    if key == '_':
                        continue
                    tags = key.split('|')
                    for tag in tags:
                        parent_tag = tag.split('=')[0]
                        child_tag = tag.split('=')[1]
                        if parent_tag not in morph_dict.keys():
                            morph_dict[parent_tag] = {}
                        if child_tag not in morph_dict[parent_tag]:
                            morph_dict[parent_tag][child_tag] = 0
                        morph_dict[parent_tag][child_tag] += int(number)
    return_values = []
    for cat in ['Animacy', 'Case', 'Gender', 'Number']:
        for subcat in morph_dict[cat]:
            total = np.sum(list(morph_dict[cat].values()))
            if total == 0:
                total = 1
            return_values.append(morph_dict[cat][subcat] /
                                 total)
    return tuple(return_values)


def generate(path, modelname, top_k):

    tokenizer, model = load_models(modelname)
    df = pd.read_csv(path, sep='\t')
    df['masked_before_target'] = df[['positions', 'context']].apply(lambda x: mask_before_target(*x), axis=1)
    df['masked_after_target'] = df[['positions', 'context']].apply(lambda x: mask_after_target(*x), axis=1)
    df['before_subst_prob'] = df['masked_before_target'].progress_apply(lambda x: predict_masked_sent(tokenizer, model, x,
                                                                                                      top_k=top_k))
    df['after_subst_prob'] = df['masked_after_target'].progress_apply(lambda x: predict_masked_sent(tokenizer, model, x,
                                                                                                    top_k=top_k))
    df['merged_subst'] = intersect_sparse(df['before_subst_prob'], df['after_subst_prob'])
    nf_cnt = get_nf_cnt(df['merged_subst'])
    topk = 128
    df['subst_texts'] = df.apply(lambda r: preprocess_substs(r.before_subst_prob[:topk],
                                                                nf_cnt=nf_cnt,
                                                                lemmatize=True),
                                                                axis=1).str.join(' ')
    out_unique = set()
    for item in df['subst_texts'].tolist():
        out_unique.update(item.split())
    with open('all_substitutions.txt', 'w') as fw:
        for word in list(out_unique):
            fw.write(word + '\n')

    # os.system("/Users/a19336136/PycharmProjects/ling_wsi/wsi_bach_thesis/gram_profiling/full_profiling_pipeline.sh all_substitutions.txt /Users/a19336136/rnc_conllu")
    # morph_profiles = json.load(open('/Users/a19336136/PycharmProjects/ling_wsi/wsi_bach_thesis/gram_profiling/output/jsons/corpus_morph.json'))
    df['Anim', 'Inan', 'Acc', 'Dat', 'Gen', 'Ins', 'Loc', 'Nom', 'Par', 'Voc',
       'Fem', 'Masc', 'Neut', 'Plur', 'Sing'] = df.progress_apply(lambda x: morph_vectors(x), axis=1)

    df.drop(columns=['before_subst_prob', 'after_subst_prob', 'merged_subst'], inplace=True)
    df.to_csv('substs_morph_profiling.csv', sep='\t')

    return df

