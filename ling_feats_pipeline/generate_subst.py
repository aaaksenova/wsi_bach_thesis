import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from tqdm.auto import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import stanza
import numpy as np
import random
from pymorphy2 import MorphAnalyzer
from collections import Counter, OrderedDict
import json
from collect_ling_stats import parse_json
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = stanza.Pipeline('ru', processors="tokenize,pos,lemma,depparse")
_ma = MorphAnalyzer()
_ma_cache = {}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_all_seeds(42)
tqdm.pandas()


def mask_before_target(idxs, line):
    """
    Gets sentence and creates masked coordination pattern before target word
    """
    start_id = int(str(idxs).split(',')[0].split('-')[0].strip())
    return line[:start_id] + '[MASK] а также ' + line[start_id:]


def mask_after_target(idxs, line):
    """
    Gets sentence and creates masked coordination pattern after target word
    """
    end_id = int(str(idxs).split(',')[0].split('-')[1].strip())
    return line[:end_id + 1] + ' а также [MASK]' + line[end_id + 1:]


def load_models(modelname):
    """
    Gets huggingface model name and uploads tokenizer and LM model
    """
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForMaskedLM.from_pretrained(modelname).to(device)
    roberta_flag = 'Roberta' in modelname
    return tokenizer, model, roberta_flag


def predict_masked_sent(tokenizer, model, text, top_k, roberta_flag=False):
    """
    Gets text and returns top_k model predictions with probabilities
    """
    # Tokenize input

    text = "[CLS] %s [SEP]" % text
    # if roberta_flag:
    #     text = text.replace("[MASK]", '<mask>')
    #     print(text)
    #     tokenized_text = tokenizer(text, truncation=True, return_tensors="pt")
    #     print(tokenized_text)
    #     masked_index = (tokenized_text.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # else:
    tokenized_text = tokenizer.tokenize(text, truncation=True)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to(device)  # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, 1500, sorted=True)

    out = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        if predicted_token.isalpha():
            token_weight = top_k_weights[i]
            out.append((token_weight.item(), predicted_token))
        if len(out) == top_k:
            return out

    print(i, len(out), 'less than', top_k)


def extract_ling_feats(idxs, text, nlp):
    """
    Extracts cfse, number and dependency relation of the target
    """
    start_id = int(idxs.split(',')[0].split('-')[0].strip())
    end_id = int(idxs.split(',')[0].split('-')[1].strip())
    processed = nlp(text)
    case = _ma.parse(text[start_id:end_id + 1])[0].tag.case
    number = _ma.parse(text[start_id:end_id + 1])[0].tag.number
    dep = ''
    for token in processed.iter_tokens():
        if start_id == token.start_char:
            dep = token.words[0].deprel
            break
    return case, number, dep


def bert_prep_vectorization(dframe, tokenizer, model):
    """
    Extracts target word prepositions and their vectors..
    """
    start_prep_idx = -1
    end_prep_idx = -1
    processed = nlp(dframe.context)
    start_id = int(dframe.positions.split(',')[0].split('-')[0].strip())
    for token in processed.iter_tokens():
        if start_id == token.start_char:
            target_idx = token.words[0].id
            for i in range(len(processed.sentences[0].tokens)):
                if processed.sentences[0].dependencies[i][0].id == target_idx \
                        and processed.sentences[0].dependencies[i][1] == 'case':
                    start_prep_idx = processed.sentences[0].dependencies[i][2].start_char
                    end_prep_idx = processed.sentences[0].dependencies[i][2].end_char
    if start_prep_idx == -1:
        return 'no_prep', None
    else:
        word = dframe.context[start_prep_idx: end_prep_idx]  # Extract word
        bert_input = tokenizer.encode(dframe.context, return_tensors="pt").to(device)  # Encode sentence
        tokenized_sent = tokenizer.tokenize(dframe.context)  # Tokenize sentence
        sent_logits = model(bert_input, return_dict=True)["last_hidden_state"]
        if word in tokenized_sent:
            # Get first instance of word:
            word_index = list(np.where(np.array(tokenized_sent) == word)[0])[0]
            word_embedding = sent_logits[:, word_index, :].cpu().detach().numpy()
        else:
            return word.lower(), None
        return word.lower(), word_embedding[0]


def bert_head_vectorization(dframe, tokenizer, model):
    """
    Extracts target word heads, their vectors, POS and dependency relation.
    """
    start_prep_idx = -1
    end_prep_idx = -1
    processed = nlp(dframe.context)
    start_id = int(dframe.positions.split(',')[0].split('-')[0].strip())
    for token in processed.iter_tokens():
        if start_id == token.start_char:
            target_idx = token.words[0].head
            for i in range(len(processed.sentences[0].tokens)):
                if processed.sentences[0].dependencies[i][0].id == target_idx and processed.sentences[0].dependencies[i][2].start_char == start_id:
                    start_prep_idx = processed.sentences[0].dependencies[i][0].start_char
                    end_prep_idx = processed.sentences[0].dependencies[i][0].end_char
                    head_pos = processed.sentences[0].dependencies[i][0].upos
                    head_deprel = processed.sentences[0].dependencies[i][0].deprel
    if end_prep_idx == -1 or end_prep_idx is None:
        return 'no_verb', None, None, None
    else:
        word = dframe.context[start_prep_idx: end_prep_idx]  # Extract word
        bert_input = tokenizer.encode(dframe.context, return_tensors="pt").to(device)  # Encode sentence
        tokenized_sent = tokenizer.tokenize(dframe.context)  # Tokenize sentence
        sent_logits = model(bert_input, return_dict=True)["last_hidden_state"]
        if word in tokenized_sent:
            # Get first instance of word:
            word_index = list(np.where(np.array(tokenized_sent) == word)[0])[0]
            word_embedding = sent_logits[:, word_index, :].cpu().detach().numpy()
        else:
            # in case the word is divided in pieces:
            prev_token = ""
            word_embedding = []
            for i, token_i in enumerate(tokenized_sent):
                token_i = token_i.lower()
                if word.startswith(token_i):
                    word_embedding.append(sent_logits[:, i, :].cpu().detach().numpy())
                    prev_token = token_i
                    continue
                if prev_token and token_i.startswith("##"):
                    word_embedding.append(sent_logits[:, i, :].cpu().detach().numpy())
                    word_embedding = np.mean(word_embedding, axis=0)
                    break
                else:
                    prev_token = ""
                    word_embedding = []
            if len(word_embedding) == 0:
                return word.lower(), None, head_pos, head_deprel
        return word.lower(), word_embedding[0], head_pos, head_deprel


def intersect_sparse(substs_probs, substs_probs_y, nmasks=1):
    """
    Combines different sets of substitutes (for different templates) using
    product of smoothed probability distributions.
    """

    vec = DictVectorizer(sparse=True)
    f1 = substs_probs.apply(lambda x: {s: p for p, s in x})
    f2 = substs_probs_y.apply(lambda x: {s: p for p, s in x})
    vec.fit(list(f1) + list(f2))
    f1, f2 = (vec.transform(list(f)) for f in (f1, f2))  # sparse matrix

    alpha1, alpha2 = ((1. - f.sum(axis=-1).reshape(-1, 1)) / 250000 ** nmasks \
                      for f in (f1, f2))
    prod = f1.multiply(f2) + f1.multiply(alpha2) + f2.multiply(alpha1)
    # + alpha1*alpha2 is ignored to preserve sparsity
    # finally, we don't want substs with 0
    # probs before smoothing in both distribs
    fn = np.array(vec.feature_names_)
    # maxlen = (substs_probs_y.apply(len) + substs_probs.apply(len)).max()
    m = prod
    n_texts = m.shape[0]

    def reverse_argsort(mdata):
        return np.argsort(mdata)[::-1]

    idx = list()
    for text_ix in range(n_texts):
        # sparce matrices are used to preserve high performance
        # refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
        # to learn the sparse matrices indexing (i.e. what is `indptr` and `data`)
        text_sparse_indices = m.indices[m.indptr[text_ix]:m.indptr[text_ix + 1]]
        text_sparse_data = m.data[m.indptr[text_ix]:m.indptr[text_ix + 1]]
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


def morph_vectors(x, morph_profiles):
    """
    Takes the line of processed dataframe and creates a vector of morphological features
    based on previously generated json
    """
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


def synt_vectors(x, synt_profiles):
    """
    Takes the line of processed dataframe and creates a vector of syntactic features
    based on previously generated json
    """
    synt_dict = OrderedDict({"acl": 0,
                             "advcl": 0,
                             "advmod": 0,
                             "amod": 0,
                             "appos": 0,
                             "aux": 0,
                             "case": 0,
                             "cc": 0,
                             "ccomp": 0,
                             "clf": 0,
                             "compound": 0,
                             "conj": 0,
                             "cop": 0,
                             "csubj": 0,
                             "dep": 0,
                             "det": 0,
                             "discourse": 0,
                             "dislocated": 0,
                             "expl": 0,
                             "fixed": 0,
                             "flat": 0,
                             "goeswith": 0,
                             "iobj": 0,
                             "list": 0,
                             "mark": 0,
                             "nmod": 0,
                             "nsubj": 0,
                             "nummod": 0,
                             "obj": 0,
                             "obl": 0,
                             "orphan": 0,
                             "parataxis": 0,
                             "punct": 0,
                             "reparandum": 0,
                             "root": 0,
                             "vocative": 0,
                             "xcomp": 0})
    synt_child_dict = OrderedDict({"acl": 0,
                                   "advcl": 0,
                                   "advmod": 0,
                                   "amod": 0,
                                   "appos": 0,
                                   "aux": 0,
                                   "case": 0,
                                   "cc": 0,
                                   "ccomp": 0,
                                   "clf": 0,
                                   "compound": 0,
                                   "conj": 0,
                                   "cop": 0,
                                   "csubj": 0,
                                   "dep": 0,
                                   "det": 0,
                                   "discourse": 0,
                                   "dislocated": 0,
                                   "expl": 0,
                                   "fixed": 0,
                                   "flat": 0,
                                   "goeswith": 0,
                                   "iobj": 0,
                                   "list": 0,
                                   "mark": 0,
                                   "nmod": 0,
                                   "nsubj": 0,
                                   "nummod": 0,
                                   "obj": 0,
                                   "obl": 0,
                                   "orphan": 0,
                                   "parataxis": 0,
                                   "punct": 0,
                                   "reparandum": 0,
                                   "root": 0,
                                   "vocative": 0,
                                   "xcomp": 0})
    substitutions = list(set(x['subst_texts'].split()))
    for word in substitutions:
        if _ma.parse(word)[0].tag.POS == 'NOUN':
            if word in synt_profiles.keys():
                for key, number in synt_profiles[word].items():
                    if key == '_':
                        continue
                    tags = key.split('|')
                    parent_tag = tags[0].split(':')[0]
                    for tag in tags[1:]:
                        child_tag = tag.split(':')[0]
                        synt_dict[parent_tag] += int(number)
                        synt_child_dict[child_tag] += int(number)
    synt_scaled = [i / max(np.sum(list(synt_dict.values())), 1) for i in list(synt_dict.values())]
    synt_child_scaled = [i / max(np.sum(list(synt_child_dict.values())), 1) for i in list(synt_child_dict.values())]
    return synt_scaled + synt_child_scaled


def generate(path, modelname, top_k, methods):
    """
    Takes path to dataset in RUSSE-18 format,
    huggingface model name and the number of substitutes to use
    """

    if os.path.exists(f"substs_profiling_{modelname.split('/')[-1]}.tsv"):
        df = pd.read_csv(f"substs_profiling_{modelname.split('/')[-1]}.tsv", sep="\t")
    else:
        tokenizer, model, roberta_flag = load_models(modelname)
        df = pd.read_csv(path, sep='\t')
        df['masked_before_target'] = df[['positions', 'context']].apply(lambda x: mask_before_target(*x), axis=1)
        df['masked_after_target'] = df[['positions', 'context']].apply(lambda x: mask_after_target(*x), axis=1)
        df[['target_case', 'target_num', 'target_dep']] = df.progress_apply(lambda x:
                                                                            extract_ling_feats(x['positions'],
                                                                                               x['context'], nlp),
                                                                            axis=1, result_type='expand')
        df = pd.get_dummies(df, columns=['target_case', 'target_num', 'target_dep'])
        df['before_subst_prob'] = df['masked_before_target'].progress_apply(lambda x: predict_masked_sent(tokenizer,
                                                                                                          model,
                                                                                                          x,
                                                                                                          top_k=top_k,
                                                                                                          roberta_flag=roberta_flag))
        df['after_subst_prob'] = df['masked_after_target'].progress_apply(lambda x: predict_masked_sent(tokenizer,
                                                                                                        model,
                                                                                                        x,
                                                                                                        top_k=top_k,
                                                                                                        roberta_flag=roberta_flag))
        df['merged_subst'] = intersect_sparse(df['before_subst_prob'], df['after_subst_prob'])
        nf_cnt = get_nf_cnt(df['merged_subst'])
        df['subst_texts'] = df.apply(lambda r: preprocess_substs(r.before_subst_prob[:top_k],
                                                                 nf_cnt=nf_cnt,
                                                                 lemmatize=True),
                                     axis=1).str.join(' ')
        out_unique = set()
        for item in df['subst_texts'].tolist():
            out_unique.update(item.split())
        with open(f"all_substitutions_{modelname.split('/')[-1]}.txt", 'w') as fw:
            for word in list(out_unique):
                fw.write(word + '\n')
        # checking if file exists
        if not os.path.exists(f"profiles/{modelname.split('/')[-1]}_morph.json"):
            print("Generating profiles")
            parse_json(f"all_substitutions_{modelname.split('/')[-1]}.txt", modelname)
            print("Generation finished")

        morph_profiles = json.load(open(
            f"profiles/{modelname.split('/')[-1]}_morph.json"))
        synt_profiles = json.load(open(
            f"profiles/{modelname.split('/')[-1]}_synt.json"))

        df[['Anim', 'Inan', 'Acc', 'Dat', 'Gen', 'Ins', 'Loc', 'Nom', 'Par', 'Voc',
            'Fem', 'Masc', 'Neut', 'Plur', 'Sing']] = df.progress_apply(lambda x: morph_vectors(x, morph_profiles), axis=1,
                                                                        result_type='expand')
        df[["acl",
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
            "xcomp",
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
            "xcomp_child"]] = df.progress_apply(lambda x: synt_vectors(x, synt_profiles), axis=1, result_type='expand')
        df.drop(columns=['before_subst_prob', 'after_subst_prob', 'merged_subst'], inplace=True)
    if 'prep' in methods:
        if not os.path.exists(f"profiles/{modelname.split('/')[-1]}_prep.npy"):
            tokenizer = AutoTokenizer.from_pretrained(modelname)
            model = AutoModel.from_pretrained(modelname).to(device)
            df[['prep', 'prep_vec']] = df.progress_apply(lambda x: bert_prep_vectorization(x, tokenizer, model),
                                                         result_type='expand', axis=1)
            df['prep_vec'] = df['prep_vec'].apply(
                lambda x: x if x is not None else np.zeros_like(df[~df['prep_vec'].isna()].iloc[0].prep_vec))
            prep_vec = df['prep_vec'].to_numpy()
            with open(f"profiles/{modelname.split('/')[-1]}_prep.npy", 'wb') as f:
                np.save(f, prep_vec)
        else:
            with open(f"profiles/{modelname.split('/')[-1]}_prep.npy", 'rb') as f:
                prep_vec = np.load(f, allow_pickle=True)
            df['prep_vec'] = prep_vec
    if 'headvec' in methods or 'headling' in methods:
        if not os.path.exists(f"profiles/{modelname.split('/')[-1]}_head.npy"):
            le = preprocessing.LabelEncoder()
            tokenizer = AutoTokenizer.from_pretrained(modelname)
            model = AutoModel.from_pretrained(modelname).to(device)
            df[['head', 'head_vec', 'head_pos', 'head_deprel']] = df.progress_apply(lambda x:
                                                                                    bert_head_vectorization(x, tokenizer,
                                                                                                            model),
                                                                                    result_type='expand', axis=1)
            df['head_vec'] = df['head_vec'].apply(
                lambda x: x if x is not None else np.zeros_like(df[~df['head_vec'].isna()].iloc[0].head_vec))
            df['head_pos'] = le.fit_transform(df['head_pos'])
            df['head_deprel'] = le.fit_transform(df['head_deprel'])
            head_vec = df['head_vec'].to_numpy()
            with open(f"profiles/{modelname.split('/')[-1]}_head.npy", 'wb') as f:
                np.save(f, head_vec)
        else:
            with open(f"profiles/{modelname.split('/')[-1]}_head.npy", 'rb') as f:
                head_vec = np.load(f, allow_pickle=True)
            df['head_vec'] = head_vec
    df.to_csv(f"substs_profiling_{modelname.split('/')[-1]}.tsv", sep='\t', index=False)
    return df
