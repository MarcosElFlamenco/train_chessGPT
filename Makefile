include secrets.mk

PYTHON := python3

TRAIN := train.py
PREPARE := data/lichess_hf_dataset/prepare.py

DATA_DIR := data/lichess_hf_dataset

LICHESS_YAML := lichess.yaml
RANDOM_YAML := random.yaml
LICHESS_FINETUNE_YAML := lichess_finetune.yaml
RANDOM_FINETUNE_YAML := random_finetune.yaml
GM_YAML := grandmaster.yaml

#LOCAL TRAINING (FOR DEBUGGING)

LOCAL_CONFIG := local.py
#for launching training
train: 
	$(PYTHON) $(TRAIN) \
		config/$(LOCAL_CONFIG)

# this script will convert a ziped csv file of chess games
# into a binary that is optimized for the model to train on
prepare:
	$(PYTHON) $(PREPARE)


#REMOTE TRAINING COMMANDS
SKY_YAML := remote_train.yaml
TRAIN_CLUSTER_NAME := train_cluster 

REMOTE_CONFIG := config/random1M.py

remote_train:
	export WANDB_API_KEY=$(WANDB_API_KEY) ENV_MODEL_CONFIG=$(REMOTE_CONFIG) && echo $$ENV_MODEL_CONFIG && sky launch -c $(TRAIN_CLUSTER_NAME) --env WANDB_API_KEY --env ENV_MODEL_CONFIG skypilot/$(SKY_YAML) -i 10 --down

stop_cluster:
	sky stop $(TRAIN_CLUSTER_NAME)

down_cluster:
	sky down $(TRAIN_CLUSTER_NAME)

remote_controller_train:
	export WANDB_API_KEY=$(WANDB_API_KEY) ENV_MODEL_CONFIG=$(REMOTE_CONFIG) && echo $$ENV_MODEL_CONFIG && sky jobs launch -c $(TRAIN_CLUSTER_NAME) --env WANDB_API_KEY --env ENV_MODEL_CONFIG skypilot/$(SKY_YAML) -i 10 --down


##LOCAL MODEL EVALUATION COMMANDS

#HELPER COMMANDS

#This command allows you to check the legal moves out of a given pgn
print_legal_moves:
	$(PYTHON) evaluation/utils/legal_moves.py


PGN_TO_EVALUATE := twic1592.pgn
STOCKFISH_LOCATION := ~/Downloads/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2

#To get a feel for stockfish valuation on real games
# outputs to the output file a list of stockfish evaluations of the configurations for a given game
# Not necessarly super valuable, plus it takes forever to run
stockfish_looks_at_gm:
	$(PYTHON) evaluation/competitive_evaluation/stockfish_looks_at_gm.py \
		--pgn_file evaluation/eval_datasets/$(PGN_TO_EVALUATE)\
		--output_file evaluation/outputs/stockfish_opinion.json \
		--max_games 2300 \
		--stockfish_path $(STOCKFISH_LOCATION)
		--time_per_move 1e-2 \


#has the model play games against stockfish at specified elo and evaluate it's resulting elo
#For more info check start of file comments
elo_evaluation_gm:
	$(PYTHON) evaluation/evaluate_elo.py \
		--models_dir ../models/random16M_finetuneGM \
		--data_dir data/lichess_hf_dataset \
		--time_per_move 1e-8 \
		--max_retries 3 \
		--evaluation_games 200 \
		--desired_elo 1320 \
		--save_file evaluation/outputs/elo_results.json \
		--save_dir elo_results \
		--stockfish_path ~/Downloads/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2 \
		--beam_width 3 \
		--evaluation_key pawn5 \
		--beam_search \
		--verbose

##RANDOM GENERATED 80 MOVES
INPUT_PGN := ';1.c3 c6 2.e3 Nf6 3.Ba6 Nd5 4.Bd3 f5 5.f4 Nxe3 6.b4 e6 7.c4 Nc2+ 8.Kf1 Bxb4 9.Nh3 Nxa1 10.Bb2 b6 11.Ng5 Bf8 12.Be5 Ba6 13.Qh5+ Ke7 14.Bc3 Bb7 15.Kg1 e5 16.Bc2 exf4 17.Qg6 Nb3 18.Na3 f3 19.Qe6+ dxe6 20.Bf6+ Kd7 21.Be7 Ba6 22.Bb1 Na5 23.Nxf3 Bxe7 24.Ne1 Kc7 25.Bxf5 Qxd2 26.Bh3 Qd5 27.g3 Bb4 28.Bg2 Bxa3 29.g4 h6 30.Bxd5 cxd5 31.Nf3 Rf8 32.h4 b5 33.g5 Be7 34.gxh6 Kd6 35.Rh3 Rh8 36.Ne1 Kc7 37.Rg3 Rxh6 38.Ra3 Bxa3 39.Kf1 Kb6 40.Nc2 bxc4 41.Ne3 Bb4 42.Nxc4+ Bxc4+ 43.Kg2 Be7 44.Kh3 Ba6 45.Kh2 Rh5 46.Kg3 Ba3 47.Kg4 Kb7 48.Kh3 Be7 49.Kh2 d4 50.a3 Rh6 51.Kh3 g5 52.a4 d3 53.Kg3 Bc5 54.Kh2 Bd6+ 55.Kh1 Bh2 56.Kg2 d2 57.Kh3 gxh4 58.Kg4 Nc4 59.Kg5 Bb5 60.Kxh6 Ba6 61.Kg6 Nd7 62.Kh7 Bc7 63.Kg7 Kb6 64.Kf7 Nd6+ 65.Kxe6 Nc8 66.Kxd7 d1=R+ 67.Ke6 Ne7 68.Kf6 Rf8+ 69.Kxe7 Bd3 70.Kd7 Rd2 71.a5+ Ka6 72.Ke7 Rh2 73.Kd7 Rh3 74.Ke6 Bxa5 75.Ke7 Kb7 76.Kd7 Re8 77.Kxe8 Bc3 78.Ke7 Rh1 79.Kd8 Ba1 80.Ke7 Rd1 81.Kd7 Rh1 82.Ke7 Bb2 83.Kf'
TEMPERATURE := 1.0
CHECKPOINT := ../models/bigrandom600/bigrandom600_15K.pth

# This will simply give the model the input PGN specified, 
# have it infer next token at each position
# and output the concatenation of each token.
# Use it to get a feel for what the model predicts
# and how to make it predict
generate_vanilla:
	$(PYTHON) evaluation/model_generate.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR)

# Has the model play against itself for entire moves (multiple tokens)
# and validates the validity of said moves in the board configuration
# The code can serve as example for generating token sequences
generate_moves:
	$(PYTHON) evaluation/generate_and_validate.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR) \
		--deterministic \
		--verbose

##BENCHMARKING
BENCHMARK_GAMES := twic1592

BENCHMARK_CSV := evaluation/eval_datasets/$(BENCHMARK_GAMES).csv
BENCHMARK_PGN := evaluation/eval_datasets/$(BENCHMARK_GAMES).pgn

GENERATE_NUM := 20
##You can generate your own random games with this command
generate_benchmark_games:
	$(PYTHON) data/random_dataset/generate_random/gen_random.py \
		--num_games $(GENERATE_NUM) \
		--output_file $(BENCHMARK_CSV)
	$(PYTHON) data/random_dataset/generate_random/toPGN.py \
		--csv_file $(BENCHMARK_CSV) \
		--pgn_file $(BENCHMARK_PGN) \
		--move_column transcript


BENCHMARK_PKL := evaluation/eval_datasets/$(BENCHMARK_GAMES).pkl
##wether the PGN comes from your generation or online real games,
##you can precompute the valid moves for every step of the game into a .pkl file
## with this command so it won't redo that math for every model that runs on the same test sample
## Also, you have to because thats what the next step takes as input
precompute_benchmark:
	$(PYTHON) evaluation/benchmark.py \
		precompute \
		--pgn_files $(BENCHMARK_PGN) \
		--output_file $(BENCHMARK_PKL) \
		--max_moves 0

RESULTS_FILE := evaluation/outputs/generation_results.csv

KARVONEN_MODEL := /home/oscar/train_ChessGPT/evaluation/eval_models/lichess_8layers_ckpt_no_optimizer.pt

EVALUATION_DATASET := evaluation/eval_datasets/$(BENCHMARK_PKL)
MODELS_DIRECTORY := ../models

## Here you can give the pkl file of your choice and benchmark a model
## By default, will evaluate all models starting with the prefix given as "models" argument
## in the "models_directory" argument
benchmark_models:
	$(PYTHON) evaluation/benchmark.py \
		eval \
		--checkpoints \
		--models_directory $(MODELS_DIRECTORY) \
		--models 2_random_600 \
		--datasets $(BENCHMARK_PKL) \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \

plot:
	$(PYTHON) evaluation/graphing_results.py

remote_benchmark_model:
	sky launch -c benchmarkCluster benchmark.yaml -i 10 --down