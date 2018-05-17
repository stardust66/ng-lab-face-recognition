import argparse
import os
from src import database

parser = argparse.ArgumentParser(description="Save embeddings of photos to"
                                             "file.")
parser.add_argument("output_path", help="The path of the output file")
parser.add_argument("input_directory", help="The directory of the photo files")
parser.add_argument("files", help="Name of photo files separated by spaces",
                    nargs="+")

if __name__ == "__main__":
    args = parser.parse_args()

    output_path = os.path.abspath(args.output_path)
    input_directory = os.path.abspath(args.input_directory)

    input_paths = [os.path.join(input_directory, filename)
                   for filename in args.files]

    database.create_database(input_paths, output_path)
