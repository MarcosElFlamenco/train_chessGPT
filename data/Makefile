PYTHON := python3
HF_REPO := MarcosElFlamenco/chess_games


PGN_DIR := twic_extracted_pgn
S3_BUCKET := go-bucket-craft
S3_DIR := dataset

MIN_ELO := 0
MAX_ELO := 30000
MIN_LEN := 30
INPUT_LENGTH := 1023
TEST_SIZE := 0.01

CURRENT_DATASET := twic_dataset

BASE := 9gb

testing:
	python3 test.py

download:
	python3 download_pgn.py \
		--pgn_directory $(PGN_DIR) \
		--zst_url $(MAR20) \
		--post_clean 1

download_twic:
	python3 download_and_upload/download_twic.py \
		--pgn_directory $(PGN_DIR) \
		--url $(TWIC_URL) \


## $(LINK_MAR24) $(LINK_APR24) $(LINK_JUN24) $(LINK_JUL24) $(LINK_AUG24) $(LINK_SEP24) \
##clean 1 means do clean and 0 means dont
	
csv:
	python3 filter/pgn_to_csv.py \
		$(CURRENT_DATASET)/$(PGN_DIR) \
		$(CURRENT_DATASET)/$(CSV_FILE) \
		--min_elo $(MIN_ELO) \
		--max_elo $(MAX_ELO) \
		--min_length $(MIN_LEN) \
		--processed_files $(CURRENT_DATASET)/processed_files.txt
		--debug

cast:
	python3 filter/castDataset.py \
		--csv_file $(CSV_FILE) \
		--dataset_name $(HF_DATASET)

LICHESS_CSV_FILE := lichess_hf_dataset/lichess_9gb.csv
LICHESS_BLOCKS_FILE := lichess_hf_dataset/lichess_9gb_blocks.csv

BASE := smartCutRandom2
CSV_FILE := random_dataset/$(BASE).csv
BLOCKS_FILE := random_dataset/$(BASE)_blocks.csv

truncate:
	python filter/truncate.py \
		--output_file $(CSV_FILE) \
		--std_dev 50

blocks:
	python3 filter/csv_to_blocks.py \
		--input_file $(CSV_FILE) \
		--input_length $(INPUT_LENGTH) \
		--csv_type quotes

tokens:
	python3 tokenizeData.py \
		--blocks_file $(BLOCKS_FILE) \
		--bin_dir $(BIN_DIR) \
		--test_size $(TEST_SIZE)


streamline: truncate blocks tokens


block_zip:
	python3 zip_all.py \
		--csv_file dummy \
		--zip_directory $(ZIP_DIR) \
		--bin_directory $(BIN_DIR) \
		--files blocks
bin_zip:
	python3 filter/zip_all.py \
		--base $(BASE) \
		--zip_directory $(ZIP_DIR) \
		--bin_directory $(BIN_DIR) \
		--files bins


stats_in_lichess:
	python3 process_datasets.py \
		train9gb.bin

uploadhf:
	python3 filter/upload_to_hf.py \
		--source_dir $(ZIP_DIR) \
		--repo_id $(HF_REPO)  \
		--token $(HF_TOKEN)

uploads3:
	python3 filter/upload_to_s3.py \
		--local_dir $(ZIP_DIR) \
		--bucket $(S3_BUCKET) \
		--s3_dir $(S3_DIR)


filter_remote: clean
	sky jobs launch -c boardCluster remote/sky.yaml

random_remote: clean
	sky jobs launch -c boardCluster remote/sky_random.yaml

hfdataset: download csv blocks block_zip uploads3 tokens bin_zip uploads3
## should be the entire pipeline for exporting a folder of pgn transcripts into a huggingface dataset
## with elos between min and max elo
	echo 'pipeline finished'

##RANDOM GENERATION
BIN_CATEGORY := forstats
RANDOM := random$(BIN_CATEGORY)
RANDOM_CSV := $(RANDOM).csv
RANDOM_PGN := $(RANDOM).pgn
RANDOM_BLOCKS := $(RANDOM)_blocks.csv
NUM_RANDOM_GAMES := 3000
RANDOM_GEN_LOG_FREQ := 1000
RANDOM_GEN_SAVE_FREQ := 50000
RANDOM_INPUT_LENGTH := 1023
ZIP_DIR := zipped
BIN_DIR := binned


random_csv:
	python3 random/gen_random.py \
		--num_games $(NUM_RANDOM_GAMES) \
		--output_file $(RANDOM_CSV) 

random_blocks:
	python3 filter/csv_to_blocks.py \
		--input_file $(RANDOM_CSV) \
		--input_length $(RANDOM_INPUT_LENGTH) \
		--csv_type quotes

random_tokens:
	python3 tokenizeData.py \
		--blocks_file $(RANDOM_BLOCKS) \
		--bin_dir $(BIN_DIR) \
		--bin_category $(BIN_CATEGORY) \
		--test_size $(TEST_SIZE)


remote_random_csv:
	python3 random/gen_random.py \
		--num_games $(NUM_RANDOM_GAMES) \
		--output_file $(RANDOM_CSV) \
		--log_freq $(RANDOM_GEN_LOG_FREQ) \
		--save_freq $(RANDOM_GEN_SAVE_FREQ) \
		--save_s3 1\
		--bucket_name $(S3_BUCKET) \
		--object_name $(RANDOM_CSV)

random_dataset: random_csv random_blocks random_tokens


random_pgn:
	python3 random/toPGN.py \
		--csv_file $(RANDOM_CSV) \
		--pgn_file $(RANDOM_PGN) \
		--move_column transcript

download_generated_random:
	python remote/download_file_from_s3.py \
		--bucket_name go-bucket-craft \
		--s3_key random.csv \
		--download_path random$(BIN_CATEGORY).csv

upload_preprocessed_random:
	python remote/upload_file_to_s3.py \
		--bucket_name bins-bucket-craft \
		--file_paths train$(BIN_CATEGORY).bin val$(BIN_CATEGORY).bin

remote_data_preprocess: clean
	sky jobs launch -c boardCluster remote/preprocess.yaml

cleanDirs:
	rm -rf zipped
	rm -rf binned

clean: cleanDirs
	rm -f $(PGN_DIR)/*
	rm -f *.csv
	rm -f train.bin val.bin