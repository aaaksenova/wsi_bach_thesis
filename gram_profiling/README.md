# Grammatic Profile Generation

The main script is `full_profiling_pipeline.sh`

The script will process your CONLL-U files and generate morphosyntax features for each target word.
Find the profiles in json format in the `output` directory.

To launch the script run

`./full_profiling_pipeline.sh TARGETS CORPUS`

- `TARGETS` is a list of target words, one per line
- `CORPUS` is a directory with a set of files,
tagged and parsed into the [CoNLL-U format](https://universaldependencies.org/format.html).