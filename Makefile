PYTHON := python3
TRAIN := train2.py
PREPARE := data/lichess_hf_dataset/prepare.py
DATA_DIR := data
CONFIG := local.py
LICHESS_YAML := lichess.yaml
RANDOM_YAML := random.yaml
LICHESS_FINETUNE_YAML := lichess_finetune.yaml
RANDOM_FINETUNE_YAML := random_finetune.yaml
GM_YAML := grandmaster.yaml

local_train: $(TRAIN) 
	$(PYTHON) $(TRAIN) \
		config/$(CONFIG)

prepare: $(PREPARE)
	$(PYTHON) $(PREPARE)

remote_controller_train_lichess: 
	sky jobs launch -c lichessGamesCluster --env WANDB_API_KEY remote/$(LICHESS_YAML)

remote_controller_train_random: 
	sky jobs launch -c randomGamesCluster --env WANDB_API_KEY remote/$(RANDOM_YAML)

remote_controller_finetune_lichess: 
	sky jobs launch -c lichessGamesCluster --env WANDB_API_KEY remote/$(LICHESS_FINETUNE_YAML)

remote_controller_finetune_random: 
	sky jobs launch -c randomGamesCluster --env WANDB_API_KEY remote/$(RANDOM_FINETUNE_YAML)


remote_nocontroller_train_lichess:
	sky launch -c boardLichessCluster --use-spot --env WANDB_API_KEY remote/$(LICHESS_YAML)

remote_nocontroller_train_gm:
	sky launch -c boardGMCluster --use-spot --env WANDB_API_KEY remote/$(GM_YAML)

remote_nocontroller_train_random:
	sky launch -c boardRandomCluster --use-spot --env WANDB_API_KEY remote/$(RANDOM_YAML)

remote_nocontroller_finetune_lichess:
	sky launch -c boardLichessCluster --use-spot --env WANDB_API_KEY remote/$(LICHESS_FINETUNE_YAML)

remote_nocontroller_finetune_random:
	sky launch -c boardRandomCluster --use-spot --env WANDB_API_KEY remote/$(RANDOM_FINETUNE_YAML)

debug_remote_nocontroller_train_random:
	sky launch -c debugCluster --use-spot --env WANDB_API_KEY remote/$(RANDOM_YAML)

control_remote_nocontroller_train_random:
	sky launch -c debugCluster2 --use-spot --env WANDB_API_KEY remote/random2.yaml


print_legal_moves:
	python3 evaluation/legal_moves.py

elo_evaluation:
	python3 evaluation/evaluate_elo.py \
		--checkpoint random_karvhypNSNR_finetune_GM_100K \
		--time_per_move 1e-8 \
		--max_retries 3 \
		--evaluation_games 200 \
		--desired_elo 1320 \
		--save_file results.pkl \
		--save_dir elo_results \
		--stockfish_path ~/Downloads/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish1 \
		--troubleshooting_verbose \
		--verbose

elo_evaluation_random:
	python3 evaluation/evaluate_elo.py \
		--models_dir ../models/random_karvhypNSNR_finetune \
		--time_per_move 1e-8 \
		--max_retries 3 \
		--evaluation_games 200 \
		--desired_elo 1320 \
		--save_file results.pkl \
		--save_dir elo_results \
		--stockfish_path ~/Downloads/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish1 \

elo_evaluation_lichess:
	python3 evaluation/evaluate_elo.py \
		--models_dir ../models/lichess_karvhyp_finetune \
		--time_per_move 1e-8 \
		--max_retries 3 \
		--evaluation_games 200 \
		--desired_elo 1320 \
		--save_file results.pkl \
		--save_dir elo_results \
		--stockfish_path ~/Downloads/stockfish-ubuntu-x86-64-avx2/stockfish/stockfish1 \


##RANDOM GENERATED 80 MOVES
INPUT_PGN := ';1.c3 c6 2.e3 Nf6 3.Ba6 Nd5 4.Bd3 f5 5.f4 Nxe3 6.b4 e6 7.c4 Nc2+ 8.Kf1 Bxb4 9.Nh3 Nxa1 10.Bb2 b6 11.Ng5 Bf8 12.Be5 Ba6 13.Qh5+ Ke7 14.Bc3 Bb7 15.Kg1 e5 16.Bc2 exf4 17.Qg6 Nb3 18.Na3 f3 19.Qe6+ dxe6 20.Bf6+ Kd7 21.Be7 Ba6 22.Bb1 Na5 23.Nxf3 Bxe7 24.Ne1 Kc7 25.Bxf5 Qxd2 26.Bh3 Qd5 27.g3 Bb4 28.Bg2 Bxa3 29.g4 h6 30.Bxd5 cxd5 31.Nf3 Rf8 32.h4 b5 33.g5 Be7 34.gxh6 Kd6 35.Rh3 Rh8 36.Ne1 Kc7 37.Rg3 Rxh6 38.Ra3 Bxa3 39.Kf1 Kb6 40.Nc2 bxc4 41.Ne3 Bb4 42.Nxc4+ Bxc4+ 43.Kg2 Be7 44.Kh3 Ba6 45.Kh2 Rh5 46.Kg3 Ba3 47.Kg4 Kb7 48.Kh3 Be7 49.Kh2 d4 50.a3 Rh6 51.Kh3 g5 52.a4 d3 53.Kg3 Bc5 54.Kh2 Bd6+ 55.Kh1 Bh2 56.Kg2 d2 57.Kh3 gxh4 58.Kg4 Nc4 59.Kg5 Bb5 60.Kxh6 Ba6 61.Kg6 Nd7 62.Kh7 Bc7 63.Kg7 Kb6 64.Kf7 Nd6+ 65.Kxe6 Nc8 66.Kxd7 d1=R+ 67.Ke6 Ne7 68.Kf6 Rf8+ 69.Kxe7 Bd3 70.Kd7 Rd2 71.a5+ Ka6 72.Ke7 Rh2 73.Kd7 Rh3 74.Ke6 Bxa5 75.Ke7 Kb7 76.Kd7 Re8 77.Kxe8 Bc3 78.Ke7 Rh1 79.Kd8 Ba1 80.Ke7 Rd1 81.Kd7 Rh1 82.Ke7 Bb2 83.Kf'
TEMPERATURE := 1.0

generate_vanilla:
	$(PYTHON) evaluation/model_generate.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR)

generate_moves:
	$(PYTHON) evaluation/gen_and_val_noS.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR) \
		--temperature $(TEMPERATURE) \
	
generate_deterministic_moves:
	$(PYTHON) evaluation/gen_and_val_S.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR) \
		--deterministic \

##BENCHMARKING
BENCHMARK_GAMES := random2128games
GENERATE_NUM := 2128
BENCHMARK_CSV := evaluation/eval_datasets/$(BENCHMARK_GAMES).csv
BENCHMARK_PGN := evaluation/eval_datasets/$(BENCHMARK_GAMES).pgn
BENCHMARK_PRECOMPUTE := evaluation/eval_datasets/$(BENCHMARK_GAMES).pkl
BENCHMARK := benchmark_full_info.py
RESULTS_FILE := evaluation/generation_results.csv

Mkarvonen = /home/oscar/train_ChessGPT/evaluation/eval_models/lichess_8layers_ckpt_no_optimizer.pt

D1 := evaluation/eval_datasets/random100games.pkl
D2 := evaluation/eval_datasets/lichess13_100g_180m.pkl
D3 := evaluation/eval_datasets/kasparov2128games.pkl
D4 := evaluation/eval_datasets/random2128games.pkl

MODELS := ../models
benchmark_models:
	$(PYTHON) evaluation/$(BENCHMARK) \
		eval \
		--checkpoints \
		--models_directory $(MODELS) \
		--models lichess_karvhyp random_karvhypNSNR \
		--datasets $(D3) $(D4) \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \

plot:
	$(PYTHON) evaluation/graphing_results.py

precompute_benchmark:
	$(PYTHON) evaluation/$(BENCHMARK) \
		precompute \
		--pgn_files $(BENCHMARK_PGN) \
		--output_file $(BENCHMARK_PRECOMPUTE) \
		--max_moves -1


generate_benchmark_games:
	$(PYTHON) data/random_dataset/generate_random/gen_random.py \
		--num_games $(GENERATE_NUM) \
		--output_file $(BENCHMARK_CSV)
	$(PYTHON) data/random_dataset/generate_random/toPGN.py \
		--csv_file $(BENCHMARK_CSV) \
		--pgn_file $(BENCHMARK_PGN) \
		--move_column transcript

generate_precompute_random_benchmark_games: generate_benchmark_games precompute_benchmark

full_benchmark: generate_precompute_random_benchmark_games benchmark_model

remote_benchmark_model:
	sky jobs launch -c benchmarkCluster remote/benchmark.yaml
	
hfdataset:
	sky jobs launch -c boardCluster remote/hfExport.yaml

scrap_chess:
	python data/automate.py \
		--url https://www.chess.com/games \
		--download_dir grandmaster_pgns \
		--demo \
		--demo_limit 3
##REMOTE
query_spot_prices:
	aws ec2 describe-spot-price-history \
	--region eu-west-2 \
	--start-time 2024-11-02T11:53:15Z \
	--instance-types g5.4xlarge g5.xlarge g5.2xlarge g5.8xlarge g5.16xlarge \
	--output table

check_spot_history:
	aws ec2 describe-spot-price-history \
	--start-time 2024-12-06T09:39:53Z \
	--instance-types g5.4xlarge g5.xlarge g5.2xlarge g5.8xlarge g5.16xlarge \
	--availability-zone eu-west-3

