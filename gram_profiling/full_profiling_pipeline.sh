#! /bin/bash

# First, we read a CONLL file and dumps frequencies
# for morphological and syntax properties
# of the target words into JSON files

TARGET=${1}  # List of target words, one per line
CONLL_DIR=${2}  # Directory with conllu.gz files

OUTJSONS=output_wiki/jsons
mkdir -p ${OUTJSONS}

echo "Extracting grammatical profiles..."
python3 collect_ling_stats_child.py -i ${CONLL_DIR} -t ${TARGET} -o ${OUTJSONS}/corpus #&
echo "Done extracting grammatical profiles"

