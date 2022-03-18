import argparse
from gram_profile_clustering import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='Arguments for WSI with grammatical profiles')
    parser.add_argument('methods', type=str, help='Write methods separated by _')
    args = parser.parse_args()
    run_pipeline(args.methods)


if __name__ == '__main__':
    main()