import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from generate_profiles import generate
from tqdm.auto import tqdm
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity
from ruwordnet import RuWordNet
from ruword_frequency import Frequency
from transformers import AutoTokenizer, AutoModel, BertConfig
import os


config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("bert-base-cased", config=config)
freq = Frequency()
freq.load()
wn = RuWordNet()

target_fname = 'rudsi'
train_words_fname = 'bnc_contexts_train'


def get_num_senses_by_profiling(target_fname, train_words_fname):
    generate(f"{train_words_fname}.tsv", train_words_fname)
    df_train = pd.read_csv(f"profiled_{train_words_fname}.tsv", sep='\t')  # Averaged num of senses
    df = pd.read_csv(f"{target_fname}.tsv", sep='\t')
    df_test = df.groupby('word').gold_sense_id.nunique().reset_index()
    df_test.rename(columns={'word': 'Lemma'}, inplace=True)
    df_test.to_csv(f"{target_fname}_unique.tsv", sep='\t', index=False)
    generate(f"{target_fname}_unique.tsv", target_fname)
    df_train = df_train.loc[~df_train.Lemma.isin(set(df_test.Lemma.tolist()))]
    df_train.to_csv(f"profiled_{train_words_fname}.tsv", sep='\t', index=False)
    X, y = df_train.loc[:, 'Anim': 'xcomp_child'].to_numpy(), df_train.Mean.to_numpy()
    parameters = [
                    {
                        'sgd_reg__max_iter': [100000, 1000000],
                        'sgd_reg__tol': [1e-10, 1e-3],
                        'sgd_reg__eta0': [0.001, 0.01]
                    }
                ]
    reg = GridSearchCV(SGDRegressor(), parameters, n_jobs=4)
    reg.fit(X=X, y=y)
    tree_model = reg.best_estimator_
    df_test = pd.read_csv(f"profiled_{target_fname}.tsv", sep='\t')
    df_test.drop(columns=['gold_sense_id'], inplace=True)
    df_test['num_senses'] = tree_model.predict(df_test.loc[:, 'Anim': 'xcomp_child'].to_numpy())
    df_test['num_senses'] = df_test['num_senses'].astype('int64')
    df = df.merge(df_test, left_on='word', right_on='Lemma')
    df.drop(columns=['Lemma'], inplace=True)
    df.to_csv(f'num_senses_ling_{target_fname}.tsv', sep='\t', index=False)


def get_idxs(word, sent):
    """
    Function gets word and sentence as an input
    and returns list of token indexes
    """
    idx = 0
    enc = [tokenizer.encode(x) for x in sent.split()]
    desired_output = []
    for token in enc:
        tokenoutput = []
        for ids in token:
            tokenoutput.append(idx)
            idx += 1
        desired_output.append(tokenoutput)
    try:
        out = desired_output[sent.split().index(word.strip())]
    except:
        out = desired_output[sent.split().index(word.strip()[:-1])]
    return out


def BERT_TOKENS(idxs, lines, model, tokenizer):
    """
    Function extracts words vectors
    as outputs from the last layer
    """

    embs = []
    for idx, line in tqdm(zip(idxs, lines), total=len(lines)):
        start_id = int(idx.split(',')[0].split('-')[0].strip())
        end_id = int(idx.split(',')[0].split('-')[1].strip())
        word = line[start_id : end_id]
        input = tokenizer.encode(line, return_tensors="pt")
        tokenized_sent = tokenizer.tokenize(line)
        sent_logits = model(input, return_dict=True)["last_hidden_state"]
        if word in tokenized_sent:
            word_index = list(np.where(np.array(tokenized_sent) == word)[0])[0]
            # print(word_index, sent_logits.size())
            word_embedding = sent_logits[:, word_index, :].cpu().detach().numpy()
            embs.append(word_embedding)
        else:
            # in case the word is divided in more than 2 pieces:
            splitted_tokens = []
            splitted_array = np.zeros((1, 768))
            prev_token = ""
            prev_array = np.zeros((1, 768))
            for i, token_i in enumerate(tokenized_sent):
                word_embedding = splitted_array = np.zeros((1, 768))
                array = sent_logits[:, i, :].cpu().detach().numpy()
                token_i = token_i.lower()

                # Find the word divided into several byte-pair encodings
                if token_i.startswith('##'):

                    if prev_token:
                        splitted_tokens.append(prev_token)
                        prev_token = ""
                        splitted_array = prev_array

                    splitted_tokens.append(token_i)
                    splitted_array += array

                else:
                    if splitted_tokens:
                        sarray = splitted_array/len(splitted_tokens)
                        stoken_i = "".join(splitted_tokens).replace('##', '')
                        if stoken_i.lower() == word:
                            # replace in the tokenized sentence the divided word by the full word
                            # tokenized_sent[i-len(splitted_tokens):i] = word
                            word_embedding = np.array(sarray)
                            # keep all the embs of the sentence, to select only one randomly
                        splitted_tokens = []
                        splitted_array = np.zeros((1, 768))
                    prev_array = array
                    prev_token = token_i
            embs.append(word_embedding)

    return np.array(embs)


def variance_counter(df):
    """
    Function calculates the variance for each word vectors distribution
    extracted from the dataset
    """

    word_dict = {}
    y = []
    print('Dataset includes {} words'.format(len(df.lemma.unique())))
    # Collect embeddings and cluster tags
    for i, lemma in enumerate(df.lemma.unique()):
        # print(word)
        word_dict[lemma] = BERT_TOKENS(df[df.lemma == lemma]['indexes_target_token'].tolist(),
                                       df[df.lemma == lemma]['context'].tolist(), model, tokenizer)
        y += [i] * word_dict[lemma].shape[0]
    # Get cluster centroids by cosine distance estimation
    clf = NearestCentroid(metric='cosine')
    x = np.concatenate([*word_dict.values()]).squeeze()
    y = np.array(y)
    centroids = clf.fit(x, y).centroids_
    variances = {lemma: np.sum(cosine_similarity(word_dict[lemma].squeeze(), centroids[0].reshape(1, -1)) ** 2) /
                        word_dict[lemma].shape[0] \
                 for i, lemma in enumerate(word_dict.keys())}
    selfsim = {lemma: np.sum(cosine_similarity(word_dict[lemma].squeeze(), word_dict[lemma].squeeze())) \
               for i, lemma in enumerate(word_dict.keys())}

    return variances, selfsim


def collect_data(filename):
    """
    The function transforms RUSSE dataset into list of tuples
    where for each word its frequency, vector variation and number of senses are calculated
    """

    df = pd.read_csv(filename, sep='\t')
    df.rename(columns={'word':'lemma', 'positions':'indexes_target_token'}, inplace=True)
    var_dict, selfsim_dict = variance_counter(df)
    freq_dict = {word: freq.ipm(word) for word in var_dict.keys()}
    num_sense_dict = {word: len(wn.get_senses(word)) for word in var_dict.keys()}
    data_list = []
    for word in var_dict.keys():
        data_list.append((word, var_dict[word], selfsim_dict[word], freq_dict[word], num_sense_dict[word]))
    return data_list


def get_num_senses_by_variance(target_fname, train_words_fname):
    df = pd.read_csv(f"{target_fname}.tsv", sep='\t')
    if not os.path.exists(f"variances_{train_words_fname}.tsv"):
        full_data = collect_data(f"{target_fname}.tsv")
        df_train = pd.DataFrame(full_data, columns=['words', 'variation', 'selfsim', 'frequency', 'num_senses'])
        df_test = collect_data(f"{train_words_fname}.tsv")
        df_test = pd.DataFrame(df_test, columns=['words', 'variation', 'selfsim', 'frequency', 'num_senses'])
        df_train = df_train.loc[~df_train.words.isin(set(df_test.words.tolist()))]
        df_train.to_csv(f'variances_{train_words_fname}.tsv', sep='\t', index=False)
        df_test.to_csv(f'variances_{target_fname}.tsv', sep='\t', index=False)
    else:
        df_train = pd.read_csv(f'variances_{train_words_fname}.tsv', sep='\t')
        df_test = pd.read_csv(f'variances_{target_fname}.tsv', sep='\t')
    X, y = df_train.loc[:, 'variation': 'frequency'].to_numpy(), df_train.num_senses.to_numpy()
    parameters = [
                {
                    'sgd_reg__max_iter':[100000, 1000000],
                    'sgd_reg__tol':[1e-10, 1e-3],
                    'sgd_reg__eta0':[0.001, 0.01]
                }
            ]
    reg = GridSearchCV(SGDRegressor(), parameters, n_jobs=4)
    reg.fit(X=X, y=y)
    best_model = reg.best_estimator_
    df_test['num_senses'] = best_model.predict(df_test.loc[:, 'variation': 'frequency'].to_numpy())
    df_test['num_senses'] = df_test['num_senses'].astype('int64')
    df = df.merge(df_test[['words', 'num_senses']], left_on='word', right_on='words')
    df.drop(columns=['words'], inplace=True)
    df.to_csv(f'num_senses_dist_{target_fname}.tsv', sep='\t', index=False)


def get_num_senses_joined_methods(target_fname, train_words_fname):
    df = pd.read_csv(f'{target_fname}.tsv', sep='\t')
    df_train_var = pd.read_csv(f"variances_{train_words_fname}.tsv", sep='\t')
    df_test_var = pd.read_csv(f'variances_{target_fname}.tsv', sep='\t')
    df_train_prof = pd.read_csv(f"profiled_{train_words_fname}.tsv", sep='\t')
    df_test_prof = pd.read_csv(f'profiled_{target_fname}.tsv', sep='\t')
    X_0, y_0 = df_train_var.loc[:, 'variation': 'frequency'].to_numpy(), df_train_var.num_senses.to_numpy()
    X_1, y_1 = df_train_prof.loc[:, 'Anim': 'nummod_child'].to_numpy(), df_train_prof.Mean.to_numpy()
    X = np.hstack((X_0, X_1))
    parameters = [
                    {
                        'sgd_reg__max_iter': [100000, 1000000],
                        'sgd_reg__tol': [1e-10, 1e-3],
                        'sgd_reg__eta0': [0.001, 0.01]
                    }
                ]
    reg = GridSearchCV(SGDRegressor(), parameters, n_jobs=4)
    reg.fit(X=X, y=y_0)
    best_model = reg.best_estimator_
    df_test_var.rename(columns={'words': 'Lemma'}, inplace=True)
    df_test_prof.drop(columns=['gold_sense_id'], inplace=True)
    df_test = df_test_var.merge(df_test_prof, on='Lemma')
    df_test.to_csv(f'test_set_{target_fname}.csv')
    df_test['num_senses'] = best_model.predict(np.hstack((df_test.loc[:, 'variation':'frequency'].to_numpy(),
                                                          df_test.loc[:, 'Anim': 'nummod_child'].to_numpy())))
    df_test['num_senses'] = df_test['num_senses'].astype('int64')
    df = df.merge(df_test[['Lemma', 'num_senses']], left_on='word', right_on='Lemma')
    df.drop(columns=['Lemma'], inplace=True)
    df.to_csv(f'num_senses_joined_{target_fname}.tsv', sep='\t', index=False)


get_num_senses_by_profiling()
get_num_senses_by_variance()
get_num_senses_joined_methods()


