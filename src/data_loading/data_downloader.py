import dload
import argparse

"""
Simply downloads and extracts the zip file.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", help="output path for the ZIP download", default="temp_data/raw/")
    args = parser.parse_args()
    dload.save_unzip("http://ergast.com/downloads/f1db_csv.zip", args.output_path, delete_after=True)
