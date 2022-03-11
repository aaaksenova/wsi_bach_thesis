from subst_clustering import run_pipeline
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Arguments for WSI with lexical substitutions')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('model', type=str, help='Path to model or name of huggingface BERT')
    parser.add_argument('top_k', type=int, help='Number of substitution used')
    args = parser.parse_args()
    if not os.path.exists(args.dataset):
        print(f"No such file or directory: {args.dataset}", file=sys.stderr)
        exit(-1)

    run_pipeline(args.dataset, args.model, args.top_k)


if __name__ == '__main__':
    main()
