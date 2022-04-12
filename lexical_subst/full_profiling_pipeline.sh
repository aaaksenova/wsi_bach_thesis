#! /bin/bash

# First, we read a CONLL file and dumps frequencies
# for morphological and syntax properties
# of the target words into JSON files


echo "Starting sbert..."
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst_morph'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_morph'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst_morph_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst_morph_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst_morph_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_subst_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_morph_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_morph_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_morph_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv sberbank-ai/sbert_large_nlu_ru 150 'ling_child'
echo "Done with sbert"

echo "Starting rubert"
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst_morph'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst_morph_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst_morph_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst_morph_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_subst_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_morph'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_morph_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_morph_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_morph_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_synt'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_synt_child'
python3 main.py ../russe-wsi-kit/data/main/bts-rnc/train.csv cointegrated/rubert-tiny 150 'ling_child'
echo "Done with rubert"

# Now, we produce separate change predictions based on morphological and syntactic profiles

#echo "Producing morphological predictions..."
#python3 compare_ling.py --input1 ${OUTJSONS}/corpus0_morph.json --input2 ${OUTJSONS}/corpus1_morph.json --output ${OUTSEPARATE}/morph --filtering 5 --separation 2step
