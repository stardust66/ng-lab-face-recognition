import argparse
import os
from src import database

parser = argparse.ArgumentParser(description="Save embeddings of photos to"
                                             "file.")
parser.add_argument("output_path", help="The path of the output file")
parser.add_argument("input_directory", help="The directory of the photo files")
parser.add_argument("files", help="Name of photo files separated by spaces"
                    ", can be 'all'", nargs="+")
parser.add_argument("--use_fixed_standardization", action="store_true")

model_path = os.path.join(
    os.path.dirname(__file__),
    "saved_models/model_vggface2"
)

if __name__ == "__main__":
    args = parser.parse_args()

    output_path = os.path.abspath(args.output_path)
    input_directory = os.path.abspath(args.input_directory)

    if args.files[0] == "all":
        files = os.listdir(input_directory)
    else:
        files = args.files

    input_paths = [os.path.join(input_directory, filename)
                   for filename in files]

    database.create_database(input_paths, output_path, model_path,
                             args.use_fixed_standardization)
