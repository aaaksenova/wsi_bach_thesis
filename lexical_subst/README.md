# Lexical substitutions generation pipeline

In order to generate and cluster lexical substitutions clone this repo and run the following command

`python3 main_emb.py [path to dataset] [huggingface model name] [number of substitutions]`

This script works with datasets in RUSSE-18 format and applies BERT-based models for which BertTokenizer and BertModel methods are defined. Resulting ARI, Silhouette Scores and number of clusters are saved in "result" directory.  


To cluster grammatical profiles run the following command

`python3 main_profiles.py [set of methods separated by _]`

You can choose between 3 methods of grammatic profile generation:

- 'morph' - morphological profiles
- 'synt' - syntactic profiles of target word
- 'child' - syntactic profiles of children dependencies

