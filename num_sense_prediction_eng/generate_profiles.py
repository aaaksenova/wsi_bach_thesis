import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import json
from collect_ling_stats import parse_json
# from pymorphy2 import MorphAnalyzer
import stanza
import os
from tqdm.auto import tqdm


tqdm.pandas()
# _ma = MorphAnalyzer()
_ma_cache = {}
nlp = stanza.Pipeline('en', processors="tokenize,pos,lemma,depparse")


def morph_vectors(x, morph_profiles):
    """
    Takes the line of processed dataframe and creates a vector of morphological features
    based on previously generated json
    """
    substitutions = [x['lemma']]
    morph_dict = {'Animacy': {'Anim': 0, 'Inan': 0},
                  'Case': {'Acc': 0, 'Dat': 0, 'Gen': 0, 'Ins': 0, 'Loc': 0,
                           'Nom': 0, 'Par': 0, 'Voc': 0},
                  'Gender': {'Fem': 0, 'Masc': 0, 'Neut': 0},
                  'Number': {'Plur': 0, 'Sing': 0}}
    for word in substitutions:
        if nlp(word).sentences[0].words[0].upos == 'NOUN':
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
            return_values.append(morph_dict[cat][subcat] / total)
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
    substitutions = [x['lemma']]
    for word in substitutions:
        if nlp(word).sentences[0].words[0].upos == 'NOUN':
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


def generate(path, name):
    print(name)
    df = pd.read_csv(path, sep='\t')
    words = df['lemma'].unique().tolist()
    if not os.path.exists(f"{name}_for_profiling.txt"):
        with open(f"{name}_for_profiling.txt", 'w') as fw:
            for word in words:
                fw.write(word+"\n")

    if not os.path.exists(f"profiles/{name}_morph.json"):
        print("Generating profiles")
        parse_json(f"{name}_for_profiling.txt", name)
        print("Generation finished")

    morph_profiles = json.load(open(
        f"profiles/{name}_morph.json"))
    synt_profiles = json.load(open(
        f"profiles/{name}_synt.json"))

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

    df.to_csv(f"profiled_{name}.tsv", sep='\t', index=False)


# generate("/Users/a19336136/PycharmProjects/ling_wsi/wsi_bach_thesis/num_sense_prediction/senses_rnc_wiki.tsv", "words")
