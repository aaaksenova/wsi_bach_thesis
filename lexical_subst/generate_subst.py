import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm.auto import tqdm
from sklearn.feature_extraction import DictVectorizer
import numpy as np


tqdm.pandas()


def mask_before_target(idxs, line):
    start_id = int(idxs.split(',')[0].split('-')[0].strip())
    return line[:start_id] + '[MASK] а также ' + line[start_id:]


def mask_after_target(idxs, line):
    end_id = int(idxs.split(',')[0].split('-')[1].strip())
    return line[:end_id + 1] + ' а также [MASK]' + line[end_id + 1:]


def load_models(modelname):
    tokenizer = BertTokenizer.from_pretrained(modelname)
    model = BertForMaskedLM.from_pretrained(modelname)
    return tokenizer, model


def predict_masked_sent(tokenizer, model, text, top_k=5):
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
    Combines different sets of substitutes (for different tamplates) using
    product of smoothed probability distributions.
    """

    vec = DictVectorizer(sparse=True)
    f1 = substs_probs.apply(lambda l: {s:p for p,s in l})
    f2 = substs_probs_y.apply(lambda l: {s:p for p,s in l})
    vec.fit(list(f1)+list(f2))
    f1, f2 = (vec.transform(list(f)) for f in (f1,f2)) # sparse matrix

    alpha1, alpha2 = ((1. - f.sum(axis=-1).reshape(-1,1)) / 250000**nmasks \
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
        l.append(good_substs)
    return l


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

    return df
