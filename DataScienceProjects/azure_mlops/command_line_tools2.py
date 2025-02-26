import argparse

# Create a parser object
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--verbosity', action='store_true')
args = parser.parse_args()

if args.verbosity:
    print("Verbose mode enabled")