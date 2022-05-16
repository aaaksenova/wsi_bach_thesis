import argparse
from cluster import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='Arguments for WSI with grammatical profiles')
    parser.add_argument('path', type=str, help='Write path to your data')
    parser.add_argument('model', type=str, help='Write huggingface bert name')
    parser.add_argument('top_k', type=str, help='Write number of substitutes')
    parser.add_argument('methods', type=str, help='Write methods separated by _')
    args = parser.parse_args()
    run_pipeline(args.path, args.model, int(args.top_k), args.methods)


if __name__ == '__main__':
    main()