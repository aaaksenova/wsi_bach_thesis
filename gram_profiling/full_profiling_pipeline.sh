#! /bin/bash

# First, we read a CONLL file and dumps frequencies
# for morphological and syntax properties
# of the target words into JSON files

TARGET=${1}  # List of target words, one per line
CONLL0=${2}  # Directory with conllu.gz files

OUTJSONS=output/jsons
mkdir -p ${OUTJSONS}

OUTSEPARATE=output/tsv
mkdir -p ${OUTSEPARATE}

echo "Extracting grammatical profiles..."
python3 collect_ling_stats_child.py -i ${CONLL0} -t ${TARGET} -o ${OUTJSONS}/corpus0 #&
echo "Done extracting grammatical profiles"

