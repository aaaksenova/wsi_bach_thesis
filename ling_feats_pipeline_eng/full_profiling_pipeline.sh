#! /bin/bash

# First we read a CONLL file and dumps frequencies
# for morphological and syntax properties
# of the target words into JSON files


for MODEL in "bert-large-cased"
do
  echo "Starting $MODEL"
  for FEATURES in 'subst' 'subst_synt'
#'headling' 'ling' 'subst' 'headvec'  'morph'  'synt'  'child'  'prep' 'headvec_headling' 'headvec_ling' 'headvec_subst' 'headvec_morph' 'headvec_synt' 'headvec_child' 'headvec_prep' 'headling_ling' 'headling_subst' 'headling_morph' 'headling_synt' 'headling_child' 'headling_prep' 'ling_subst' 'ling_morph' 'ling_synt' 'ling_child' 'ling_prep' 'subst_morph' 'subst_synt' 'subst_child' 'subst_prep' 'morph_synt' 'morph_child' 'morph_prep' 'synt_child' 'synt_prep' 'child_prep' 'headvec_headling_ling' 'headvec_headling_subst' 'headvec_headling_morph' 'headvec_headling_synt' 'headvec_headling_child' 'headvec_headling_prep' 'headvec_ling_subst' 'headvec_ling_morph' 'headvec_ling_synt' 'headvec_ling_child' 'headvec_ling_prep' 'headvec_subst_morph' 'headvec_subst_synt' 'headvec_subst_child' 'headvec_subst_prep' 'headvec_morph_synt' 'headvec_morph_child' 'headvec_morph_prep' 'headvec_synt_child' 'headvec_synt_prep' 'headvec_child_prep' 'headling_ling_subst' 'headling_ling_morph' 'headling_ling_synt' 'headling_ling_child' 'headling_ling_prep' 'headling_subst_morph' 'headling_subst_synt' 'headling_subst_child' 'headling_subst_prep' 'headling_morph_synt' 'headling_morph_child' 'headling_morph_prep' 'headling_synt_child' 'headling_synt_prep' 'headling_child_prep' 'ling_subst_morph' 'ling_subst_synt' 'ling_subst_child' 'ling_subst_prep' 'ling_morph_synt' 'ling_morph_child' 'ling_morph_prep' 'ling_synt_child' 'ling_synt_prep' 'ling_child_prep' 'subst_morph_synt' 'subst_morph_child' 'subst_morph_prep' 'subst_synt_child' 'subst_synt_prep' 'subst_child_prep' 'morph_synt_child' 'morph_synt_prep' 'morph_child_prep' 'synt_child_prep' 'headvec_headling_ling_subst' 'headvec_headling_ling_morph' 'headvec_headling_ling_synt' 'headvec_headling_ling_child' 'headvec_headling_ling_prep' 'headvec_headling_subst_morph' 'headvec_headling_subst_synt' 'headvec_headling_subst_child' 'headvec_headling_subst_prep' 'headvec_headling_morph_synt' 'headvec_headling_morph_child' 'headvec_headling_morph_prep' 'headvec_headling_synt_child' 'headvec_headling_synt_prep' 'headvec_headling_child_prep' 'headvec_ling_subst_morph' 'headvec_ling_subst_synt' 'headvec_ling_subst_child' 'headvec_ling_subst_prep' 'headvec_ling_morph_synt' 'headvec_ling_morph_child' 'headvec_ling_morph_prep' 'headvec_ling_synt_child' 'headvec_ling_synt_prep' 'headvec_ling_child_prep' 'headvec_subst_morph_synt' 'headvec_subst_morph_child' 'headvec_subst_morph_prep' 'headvec_subst_synt_child' 'headvec_subst_synt_prep' 'headvec_subst_child_prep' 'headvec_morph_synt_child' 'headvec_morph_synt_prep' 'headvec_morph_child_prep' 'headvec_synt_child_prep' 'headling_ling_subst_morph' 'headling_ling_subst_synt' 'headling_ling_subst_child' 'headling_ling_subst_prep' 'headling_ling_morph_synt' 'headling_ling_morph_child' 'headling_ling_morph_prep' 'headling_ling_synt_child' 'headling_ling_synt_prep' 'headling_ling_child_prep' 'headling_subst_morph_synt' 'headling_subst_morph_child' 'headling_subst_morph_prep' 'headling_subst_synt_child' 'headling_subst_synt_prep' 'headling_subst_child_prep' 'headling_morph_synt_child' 'headling_morph_synt_prep' 'headling_morph_child_prep' 'headling_synt_child_prep' 'ling_subst_morph_synt' 'ling_subst_morph_child' 'ling_subst_morph_prep' 'ling_subst_synt_child' 'ling_subst_synt_prep' 'ling_subst_child_prep' 'ling_morph_synt_child' 'ling_morph_synt_prep' 'ling_morph_child_prep' 'ling_synt_child_prep' 'subst_morph_synt_child' 'subst_morph_synt_prep' 'subst_morph_child_prep' 'subst_synt_child_prep' 'morph_synt_child_prep' 'headvec_headling_ling_subst_morph' 'headvec_headling_ling_subst_synt' 'headvec_headling_ling_subst_child' 'headvec_headling_ling_subst_prep' 'headvec_headling_ling_morph_synt' 'headvec_headling_ling_morph_child' 'headvec_headling_ling_morph_prep' 'headvec_headling_ling_synt_child' 'headvec_headling_ling_synt_prep' 'headvec_headling_ling_child_prep' 'headvec_headling_subst_morph_synt' 'headvec_headling_subst_morph_child' 'headvec_headling_subst_morph_prep' 'headvec_headling_subst_synt_child' 'headvec_headling_subst_synt_prep' 'headvec_headling_subst_child_prep' 'headvec_headling_morph_synt_child' 'headvec_headling_morph_synt_prep' 'headvec_headling_morph_child_prep' 'headvec_headling_synt_child_prep' 'headvec_ling_subst_morph_synt' 'headvec_ling_subst_morph_child' 'headvec_ling_subst_morph_prep' 'headvec_ling_subst_synt_child' 'headvec_ling_subst_synt_prep' 'headvec_ling_subst_child_prep' 'headvec_ling_morph_synt_child' 'headvec_ling_morph_synt_prep' 'headvec_ling_morph_child_prep' 'headvec_ling_synt_child_prep' 'headvec_subst_morph_synt_child' 'headvec_subst_morph_synt_prep' 'headvec_subst_morph_child_prep' 'headvec_subst_synt_child_prep' 'headvec_morph_synt_child_prep' 'headling_ling_subst_morph_synt' 'headling_ling_subst_morph_child' 'headling_ling_subst_morph_prep' 'headling_ling_subst_synt_child' 'headling_ling_subst_synt_prep' 'headling_ling_subst_child_prep' 'headling_ling_morph_synt_child' 'headling_ling_morph_synt_prep' 'headling_ling_morph_child_prep' 'headling_ling_synt_child_prep' 'headling_subst_morph_synt_child' 'headling_subst_morph_synt_prep' 'headling_subst_morph_child_prep' 'headling_subst_synt_child_prep' 'headling_morph_synt_child_prep' 'ling_subst_morph_synt_child' 'ling_subst_morph_synt_prep' 'ling_subst_morph_child_prep' 'ling_subst_synt_child_prep' 'ling_morph_synt_child_prep' 'subst_morph_synt_child_prep' 'headvec_headling_ling_subst_morph_synt' 'headvec_headling_ling_subst_morph_child' 'headvec_headling_ling_subst_morph_prep' 'headvec_headling_ling_subst_synt_child' 'headvec_headling_ling_subst_synt_prep' 'headvec_headling_ling_subst_child_prep' 'headvec_headling_ling_morph_synt_child' 'headvec_headling_ling_morph_synt_prep' 'headvec_headling_ling_morph_child_prep' 'headvec_headling_ling_synt_child_prep' 'headvec_headling_subst_morph_synt_child' 'headvec_headling_subst_morph_synt_prep' 'headvec_headling_subst_morph_child_prep' 'headvec_headling_subst_synt_child_prep' 'headvec_headling_morph_synt_child_prep' 'headvec_ling_subst_morph_synt_child' 'headvec_ling_subst_morph_synt_prep' 'headvec_ling_subst_morph_child_prep' 'headvec_ling_subst_synt_child_prep' 'headvec_ling_morph_synt_child_prep' 'headvec_subst_morph_synt_child_prep' 'headling_ling_subst_morph_synt_child' 'headling_ling_subst_morph_synt_prep' 'headling_ling_subst_morph_child_prep' 'headling_ling_subst_synt_child_prep' 'headling_ling_morph_synt_child_prep' 'headling_subst_morph_synt_child_prep' 'ling_subst_morph_synt_child_prep' 'headvec_headling_ling_subst_morph_synt_child' 'headvec_headling_ling_subst_morph_synt_prep' 'headvec_headling_ling_subst_morph_child_prep' 'headvec_headling_ling_subst_synt_child_prep' 'headvec_headling_ling_morph_synt_child_prep' 'headvec_headling_subst_morph_synt_child_prep' 'headvec_ling_subst_morph_synt_child_prep' 'headling_ling_subst_morph_synt_child_prep' 'headvec_headling_ling_subst_morph_synt_child_prep'
  do
    python3 main.py ./num_senses_ling_dwug_old.tsv $MODEL 150 $FEATURES
    echo "$FEATURES done"
  done
  echo "Finished $MODEL"
done
python3 joint_statistics.py
