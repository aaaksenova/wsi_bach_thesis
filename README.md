# Linguistically Motivated Word Sense Induction for Russian


### Can interpretable linguistic-based features perform better than BERT embeddings?

In this repo you will find the scripts to run the evaluation of different features in Word Sense Induction task for Russian nouns.

This work deals with comparing the performance of several types of linguistic features with standard Lexical Substitution method suggested by Arefyev et. al. 2021.
We have created a pipeline for comparing the importance of 8 linguistic feature sets, namely:

* **Lexical substitute vectors** – tf-idf encoded list of context-based substitutes of the target word

* Aggregated **Morphological** and **syntactic** behavioural profiles of the substitutes based on Russian National Corpus

* Aggregated **syntactic** behavioural profiles of the substitutes' dependencies based on Russian National Corpus

* **Linguistic profile** of the target word in the given context

* Embedding of the **preposition** related to the target word

* Embedding of the **head** of the target word

* **POS and syntactic relations** of the head of the target word
