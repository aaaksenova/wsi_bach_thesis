# python3
# coding: utf-8

import argparse
import logging
from smart_open import open
import os
import json


def parse_json(target, model):
    """
    Takes file with one word per line and collects
    linguistic profiles for each of the words
    writes down the result_bts_rnc into file named by model name
    and the type of profile. Model is a name of the model that
    was used for substitutes generation
    """

    inp = '../bnc_conllu'
    target_words = {}

    for line in open(target, "r"):
        word = line.strip().split("\t")[0]
        pos = None
        if "_" in word:
            word, pos = word.split("_")
        if pos == "nn":
            pos = "NOUN"
        elif pos == "vb":
            pos = "VERB"
        target_words[word] = pos

    morph_properties = {w: {} for w in target_words}
    syntax_properties = {w: {} for w in target_words}
    dep_dict = {}
    rels = []

    for input_conllu in os.listdir(inp):
        for line in open(os.path.join(inp, input_conllu), "r"):
            if not line.strip():
                continue
            if line.startswith("# text"):
                # add ancestor dependencies to the syntactic dictionary
                for idx, rel, lemma in rels:
                    if idx in dep_dict.keys():
                        rel_set = rel + '|' + '|'.join(sorted(dep_dict[idx]))
                    else:
                        rel_set = rel
                    if rel_set not in syntax_properties[lemma]:
                        syntax_properties[lemma][rel_set] = 0
                    syntax_properties[lemma][rel_set] += 1
                rels = []
                dep_dict = {}
                continue
            if line.startswith("# "):
                continue
            (
                identifier,
                form,
                lemma,
                pos,
                xpos,
                feats,
                head,
                rel,
                enh,
                misc,
            ) = line.strip().split("\t")
            # get dictionary of ancestor dependencies
            if head not in dep_dict.keys():
                dep_dict[head] = []
            dep_dict[head].append(rel)

            if lemma in target_words:
                if target_words[lemma]:
                    if pos != target_words[lemma]:
                        continue
                if feats not in morph_properties[lemma]:
                    morph_properties[lemma][feats] = 0
                morph_properties[lemma][feats] += 1
                rels.append((identifier, rel, lemma))

        for idx, rel, lemma in rels:
            if idx in dep_dict.keys():
                rel_set = rel + '|' + '|'.join(sorted(dep_dict[idx]))
            if rel_set not in syntax_properties[lemma]:
                syntax_properties[lemma][rel_set] = 0
            syntax_properties[lemma][rel_set] += 1
        rels = []
        dep_dict = {}

        with open(f"profiles/{model.split('/')[-1]}_morph.json", "w") as f:
            out = json.dumps(morph_properties, ensure_ascii=False, indent=4)
            f.write(out)
        with open(f"profiles/{model.split('/')[-1]}_synt.json", "w") as f:
            out = json.dumps(syntax_properties, ensure_ascii=False, indent=4)
            f.write(out)
