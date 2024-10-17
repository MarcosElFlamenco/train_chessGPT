PYTHON := python3
HF_REPO := MarcosElFlamenco/chess_games
PGN_DIRECTORY := downloaded_pgn
CSV_FILE := lowerThan1000.csv
ZIP_DIR := zipped
S3_BUCKET := go-bucket-craft
S3_DIR := dataset

MIN_ELO := 0
MAX_ELO := 1300
MIN_LEN := 300
INPUT_LENGTH := 1023
LINK_OCT14 := https://database.lichess.org/standard/lichess_db_standard_rated_2014-10.pgn.zst
LINK_NOV14 := https://database.lichess.org/standard/lichess_db_standard_rated_2014-11.pgn.zst

LINK_FEB13 := https://database.lichess.org/standard/lichess_db_standard_rated_2013-02.pgn.zst 
LINK_MAR13 := https://database.lichess.org/standard/lichess_db_standard_rated_2013-03.pgn.zst
LINK_APR13 := https://database.lichess.org/standard/lichess_db_standard_rated_2013-04.pgn.zst



HF_TOKEN := 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

testing:
	python3 test.py

download:
	python3 download_pgn.py \
		--pgn_directory $(PGN_DIRECTORY) \
		--zst_url $(LINK_FEB13) $(LINK_MAR13) $(LINK_APR13) \
		--post_clean 1

##clean 1 means do clean and 0 means dont
	
csv:
	python3 pgn_to_csv.py \
		--pgn_directory $(PGN_DIRECTORY) \
		--output_csv $(CSV_FILE) \
		--min_elo $(MIN_ELO) \
		--max_elo $(MAX_ELO) \
		--min_len $(MIN_LEN) \

cast:
	python3 castDataset.py \
		--csv_file $(CSV_FILE) \
		--dataset_name $(HF_DATASET)

blocks:
	python3 csv_to_blocks.py \
		--input_file $(CSV_FILE) \
		--input_length $(INPUT_LENGTH)

zip:
	python3 zip_all.py \
		--csv_file $(CSV_FILE) \
		--zip_directory $(ZIP_DIR)

uploadhf:
	python3 upload2.py \
		--source_dir $(ZIP_DIR) \
		--repo_id $(HF_REPO)  \
		--token $(HF_TOKEN)

uploads3:
	python3 upload_to_s3.py \
		--local_dir $(ZIP_DIR) \
		--bucket $(S3_BUCKET) \
		--s3_dir $(S3_DIR)

tokens:
	python3 tokenize.py \
		--csv_file $(CSV_FILE)

plot:
	python3 elo_graph.py


move:
	cp poc_blocks.csv ~/train_ChessGPT/data/lichess_hf_dataset

remote:
	sky jobs launch -c boardCluster remote/sky.yaml

hfdataset: download csv blocks zip uploads3
## should be the entire pipeline for exporting a folder of pgn transcripts into a huggingface dataset
## with elos between min and max elo
	echo 'pipeline finished'

clean:
	rm -f $(PGN_DIRECTORY)/*
	rm -f $(ZIP_DIR)/*
	rm -f *.csv
	rm -f train.bin val.bin