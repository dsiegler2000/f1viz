cd /Users/dsiegler/PycharmProjects/F1Viz/
source venv/bin/activate
export PYTHONPATH="src/"
echo "Downloading data..."
python src/data_loading/data_downloader.py
echo "Preprocessing data..."
python src/data_loading/data_preprocessor.py --suppress_output
echo "Processing data..."
python src/data_loading/data_processor.py