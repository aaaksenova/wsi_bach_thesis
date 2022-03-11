# Lexical substitutions generation pipeline

In order to generate and cluster lexical substitutions clone this repo and run the following command

`python3 main.py [path to dataset] [huggingface model name] [number of substitutions]`

This script works with datasets in RUSSE-18 format and applies BERT-based models for which BertTokenizer and BertModel methods are defined. Resulting ARI, Silhouette Scores and number of clusters are saved in "result" directory.  