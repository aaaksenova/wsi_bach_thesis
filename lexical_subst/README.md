# Lexical substitutions generation pipeline

In order to generate and cluster lexical substitutions and/or linguistic profiles clone this repo and run the following command

`python3 main.py [path to RUSSE-18 formatted dataset] [huggingface model name] [number of substitutions] [features for clustering]`

This script works with datasets in RUSSE-18 format and applies BERT-based models for which BertTokenizer and BertModel methods are defined. Resulting ARI, Silhouette Scores and number of clusters are saved in "result" directory.  

You can choose between 4 types of features to use for clustering:

- 'subst' - lexical substitutes (Arefyev et. al. 2021)
- 'morph' - morphological profiles of substitutes in RNC
- 'synt' - syntactic profiles of substitutes in RNC
- 'child' - syntactic profiles of children dependencies of substitutes in RNC
- 'ling' - morphological and syntactic features of target word in each sentence

The features can be combined. In this case they should be passed separated by '_'

