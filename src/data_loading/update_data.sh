echo "NOTE: must be running from source root (one dir above src) for this to work"
export PYTHONPATH="src/"
python src/data_loading/data_downloader.py
echo "Preprocessing data..."
python src/data_loading/data_preprocessor.py --suppress_output
echo "Processing data..."
python src/data_loading/data_processor.py