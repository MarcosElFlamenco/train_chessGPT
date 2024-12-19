PYTHON := python3
TRAIN := train.py
PREPARE := data/lichess_hf_dataset/prepare.py
DATA_DIR := data/lichess_hf_dataset
CONFIG := local.py
LICHESS_YAML := lichess.yaml
RANDOM_YAML := random.yaml

local_train: $(TRAIN) 
	$(PYTHON) $(TRAIN) \
		config/$(CONFIG)

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


prepare: $(PREPARE)
	$(PYTHON) $(PREPARE)

remote_controller_train_lichess: 
	sky jobs launch -c lichessGamesCluster --env WANDB_API_KEY remote/$(LICHESS_YAML)

remote_controller_train_random: 
	sky jobs launch -c randomGamesCluster --env WANDB_API_KEY remote/$(RANDOM_YAML)

remote_nocontroller_train_lichess:
	sky launch -c boardLichessCluster --use-spot --env WANDB_API_KEY remote/$(LICHESS_YAML)

remote_nocontroller_train_random:
	sky launch -c boardRandomCluster --use-spot --env WANDB_API_KEY remote/$(RANDOM_YAML)


toy_remote: $(MAIN)
	sky jobs launch -c boardCluster --env WANDB_API_KEY remote/toy.yaml

dummy_remote:
	python3 train.py config/random16M.py


print_legal_moves:
	python3 evaluation/legal_moves.py

##RANDOM GENERATED 80 MOVES
INPUT_PGN := ';1. c3 c6 2. e3 Nf6 3. Ba6 Nd5 4. Bd3 f5 5. f4 Nxe3 6. b4 e6 7. c4 Nc2+ 8. Kf1 Bxb4 9. Nh3 Nxa1 10. Bb2 b6 11. Ng5 Bf8 12. Be5 Ba6 13. Qh5+ Ke7 14. Bc3 Bb7 15. Kg1 e5 16. Bc2 exf4 17. Qg6 Nb3 18. Na3 f3 19. Qe6+ dxe6 20. Bf6+ Kd7 21. Be7 Ba6 22. Bb1 Na5 23. Nxf3 Bxe7 24. Ne1 Kc7 25. Bxf5 Qxd2 26. Bh3 Qd5 27. g3 Bb4 28. Bg2 Bxa3 29. g4 h6 30. Bxd5 cxd5 31. Nf3 Rf8 32. h4 b5 33. g5 Be7 34. gxh6 Kd6 35. Rh3 Rh8 36. Ne1 Kc7 37. Rg3 Rxh6 38. Ra3 Bxa3 39. Kf1 Kb6 40. Nc2 bxc4 41. Ne3 Bb4 42. Nxc4+ Bxc4+ 43. Kg2 Be7 44. Kh3 Ba6 45. Kh2 Rh5 46. Kg3 Ba3 47. Kg4 Kb7 48. Kh3 Be7 49. Kh2 d4 50. a3 Rh6 51. Kh3 g5 52. a4 d3 53. Kg3 Bc5 54. Kh2 Bd6+ 55. Kh1 Bh2 56. Kg2 d2 57. Kh3 gxh4 58. Kg4 Nc4 59. Kg5 Bb5 60. Kxh6 Ba6 61. Kg6 Nd7 62. Kh7 Bc7 63. Kg7 Kb6 64. Kf7 Nd6+ 65. Kxe6 Nc8 66. Kxd7 d1=R+ 67. Ke6 Ne7 68. Kf6 Rf8+ 69. Kxe7 Bd3 70. Kd7 Rd2 71. a5+ Ka6 72. Ke7 Rh2 73. Kd7 Rh3 74. Ke6 Bxa5 75. Ke7 Kb7 76. Kd7 Re8 77. Kxe8 Bc3 78. Ke7 Rh1 79. Kd8 Ba1 80. Ke7 Rd1 81. Kd7 Rh1 82. Ke7 Bb2 83. Kf'
##RANDOM GENERATED 12 MOVES
#INPUT_PGN := ";1. Nf3 d6 2. c4 Qd7 3. h4 a6 4. e4 Qg4 5. b3 Qxe4+ 6. Be2 g5 7. g3 Qxh4 8. a3 Ra7 9. O-O Qe4 10. c5 f5 11. Re1 Qxb1 12. Bc4 Qb2"
##NON RANDOM 80 MOVES
#INPUT_PGN := ';1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd2 d5 6. a3 Be7 7. Nf3 c5 8. dxc5 Bxc5 9. Qc2 dxc4 10. Bxc4 Nbd7 11. Rd1 Be7 12. Ng5 h6 13. h4 Qc7 14. Be2 Rd8 15. Rc1 Nf8 16. Nge4 Nxe4 17. Nxe4 Qxc2 18. Rxc2 Bd7 19. Bb4 Bxb4+ 20. axb4 Bc6 21. Nc5 Bxg2 22. Rg1 Bd5 23. e4 Bc6 24. b5 Be8 25. Nxb7 Rd4 26. Rc4 Rd7 27. Nc5 Rc7 28. Rc3 Rac8 29. b4 Nd7 30. Rcg3 Nxc5 31. bxc5 Rxc5 32. Rxg7+ Kf8 33. Bd3 Rd8 34. Ke2 Rc3 35. Rg8+ Ke7 36. R1g3 e5 37. Rh8 Rd6 38. b6 Rxb6 39. Rxe8+ Kxe8 40. Bb5+ Rxb5 41. Rxc3 Kd7 42. Rf3 Ke7 43. Rc3 a5 44. Rc7+ Kf6 45. Rc6+ Kg7 46. Ra6 Rb2+ 47. Kf3 Ra2 48. Kg3 h5 49. Ra8 Ra1 50. Kg2 a4 51. Ra5 f6 52. Kf3 a3 53. Ra6 Kf7 54. Ke3 Ke8 55. Ke2 Ke7 56. Kf3 Ra2 57. Ke3 Ra1 58. Ke2 Kf7 59. Kf3 Ra2 60. Ke3 Ke7 61. Kf3 Kd7 62. Rxf6 Rb2 63. Ra6 Rb3+ 64. Kg2 Kc7 65. f4 exf4 66. e5 Kb7 67. Ra4 Kc6 68. Ra6+ Kb5 69. Ra7 Kb6 70. Ra8 Kc5 71. Ra6 Kb5 72. Ra7 Kb6 73. Ra8 Kc6 74. Ra6+ Kd7 75. Kf2 Ke7 76. Kg2 Re3 77. Kf2 Rg3 78. Kf1 Rc3 79. Kf2 Re3 80. Kg2 Kd7 81. Kf2 Kc7 82. e6 Kd8'
##TEST SECOND GAME

##INPUT_PGN := ";1. g4 g5 2. f4 f5 3. h4 b6 4. b4 Nh6 5. Ba3 a6 6. h5 Bg7 7. Bc1 a5 8. a4 c6 9. fxg5 Rg8 10. Rh3 Nf7 11. h6 e5 12. Rf3 b5 13. d3 Ra7 14. Rxf5 Na6 15. Qd2 d5 16. Qc3 Bxh6 17. Bh3 Qd6 18. Nd2 Nb8 19. Rb1 Qf6 20. Bb2 Qe6 21. Qxc6+ Rd7 22. Qb6 Bb7 23. Rxe5 Nxg5 24. c4 Rg6 25. Rxd5 bxc4 26. Rb5 Nxh3 27. Nxc4 Bh1 28. g5 Qf6 29. Qb7 Nxg5 30. e3 Qd6 31. d4 Qe7 32. Nb6 Rxd4 33. Nc8 Rg8 34. Ke2 Rxb4 35. Qe4 Qe5 36. Bc1 Bf3+ 37. Kd3 Ne6 38. Qxf3 Rg7 39. Qd1 Qe4+ 40. Kd2 Qxb1 41. Qf3 Bg5 42. Qf8+ Nxf8 43. Nd6+ Ke7 44. Rxg5 Qa1 45. Ke2 Qb2+ 46. Ke1 Qc2 47. Rc5 Qe2+ 48. Nxe2 Nc6 49. Rh5 Rb7 50. Rh6 Ng6 51. Kd1 Kf8 52. Nc8 Rb4 53. Nb6 Nf4 54. Rxc6 Nd3 55. Ng3 Rd4 56. Nh1 Re7 57. Rc3 Rc7 58. Bd2 Rc6 59. Ra3 Rxb6 60. Ra1 Nc1 61. Nf2 Rg4 62. Bxc1 Ke8 63. Bb2 Rbg6 64. Bh8 Rd4+ 65. Kc2 Rgg4 66. Bg7 Rdf4 67. Rc1 Rg6 68. Kc3 Kd7 69. Kd3 Rg2 70. Ng4 Rff2 71. Nxf2 Rg3 72. Rc5 Ke7 73. Kc2 Rxe3 74. Rxa5 Rg3 75. Nd1 Rg4 76. Kb2 Rxa4 77. Be5 Kf8 78. Ra8+ Rxa8 79. Bg7+ Kg8 80. Bf8 Ra6 81. Ba3 Kg7 82. Ka2 Re6 83. Kb1 Re2 84. Nc3 Re5 85. K"
##deterministic and short
#INPUT_PGN := ";1.d4 e5 2.dxe5 d6 3.exd6 Bxd6 4.Nf3 Nf6 5.Nc3 O-O 6.a3 Nc6 7.e3 a6 8.Be2 h6 9.O-O Ne5 10.Bd2 Nxf3+ 11.Bxf3 Be5 12.Rc1 c6 13.Qe2 Qd6 14.Rfd1 Bxh2+ 15.Kh1 Be5 16.e4 Bxc3 17.Bxc3 Qe6 18.Rd3 Bd7 19.Rcd1 Rad8 20.Bxf6 gxf6 21.Rd6 Qe7 22.Rd1d2 Be6 23.Rxd8 Rxd8 24.Rxd8+ Qxd8 25.c4 Qd4 26.c5 Qxc5 27.Qd2 f5 28.exf5 Bxf5 29.Qxh6 Bg6 30.Be4 Bxe4 31.Qh4 Bg6 32.Qd8+ Kg7 33.Qc7 b5 34.b4 Qc1+ 35.Kh2 Qxa3 36.Qe5+ Kg8 37.Qe8+ Kg7 38.Qxc6 Qxb4 39.Qxa6 Qh4+ 40.Kg1 b4 41.Qa1+ Qf6 42.Qa4 Qc3 43.f3 b3 44.Qa3 Qc2 45.Kh2 b2"
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
BENCHMARK_GAMES := toyBench
GENERATE_NUM := 100
BENCHMARK_CSV := evaluation/eval_datasets/$(BENCHMARK_GAMES).csv
BENCHMARK_PGN := evaluation/eval_datasets/$(BENCHMARK_GAMES).pgn
BENCHMARK_PRECOMPUTE := evaluation/eval_datasets/$(BENCHMARK_GAMES).pkl
BENCHMARK1 := benchmark1.py
BENCHMARK2 := benchmark2.py
RESULTS_FILE := evaluation/benchmark_results.csv

##ok
M1 := evaluation/eval_models/lichess9gb_8layer_2K.pth

##ok
M2 := evaluation/eval_models/lichess9gb_8layer_21K.pth
M3 := evaluation/eval_models/random16M_8layer_6K.pth
M4 := evaluation/eval_models/random16M_8layer_12K.pth
M5 := evaluation/eval_models/random16M_8layer_22K.pth

#ok
M6 := evaluation/eval_models/lichess9gb_8layer_30K.pth
M7 := evaluation/eval_models/lichess9gb_8layer_40K.pth
M8 := evaluation/eval_models/lichess9gb_8layer_50K.pth
M9 := evaluation/eval_models/lichess9gb_8layer_60K.pth
M0 := evaluation/eval_models/lichess9gb_8layer_70K.pth
Ma := evaluation/eval_models/lichess9gb_8layer_80K.pth
Mb := evaluation/eval_models/lichess9gb_8layer_90K.pth
Mc := evaluation/eval_models/lichess9gb_8layer_100K.pth
Md := evaluation/eval_models/random16M_8layer_33K.pth
Me := evaluation/eval_models/random16M_8layer_40K.pth
Mf := evaluation/eval_models/random16M_8layer_50K.pth
Mg := evaluation/eval_models/random16M_8layer_60K.pth
Mh := evaluation/eval_models/random16M_8layer_70K.pth
Mi := evaluation/eval_models/random16M_8layer_80K.pth
Mj := evaluation/eval_models/random16M_8layer_90K.pth
Mk := evaluation/eval_models/random16M_8layer_100K.pth
Ml := evaluation/eval_models/lichess9gb_8layer_110K.pth
Mm := evaluation/eval_models/lichess9gb_8layer_120K.pth
Mn := evaluation/eval_models/lichess9gb_8layer_130K.pth
Mo := evaluation/eval_models/lichess9gb_8layer_140K.pth
Mp := evaluation/eval_models/lichess9gb_8layer_150K.pth
Mq := evaluation/eval_models/lichess9gb_8layer_160K.pth
Mr := evaluation/eval_models/lichess9gb_8layer_170K.pth

##a tester

Ms := evaluation/eval_models/lichess9gb_8layer_180K.pth



D1 := evaluation/eval_datasets/random100games.pkl
D2 := evaluation/eval_datasets/lichess13_100g_180m.pkl

benchmark_model:
	$(PYTHON) evaluation/$(BENCHMARK2) \
		eval \
		--checkpoints $(Mm) \
		--datasets $(D1) $(D2) \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \


plot:
	$(PYTHON) evaluation/graphing_results.py

precompute_benchmark:
	$(PYTHON) evaluation/$(BENCHMARK1) \
		precompute \
		--pgn_files $(BENCHMARK_PGN) \
		--output_file $(BENCHMARK_PRECOMPUTE) \


generate_benchmark_games:
	$(PYTHON) data/lichess_hf_dataset/random/gen_random.py \
		--num_games $(GENERATE_NUM) \
		--output_file $(BENCHMARK_CSV)
	$(PYTHON) data/lichess_hf_dataset/random/toPGN.py \
		--csv_file $(BENCHMARK_CSV) \
		--pgn_file $(BENCHMARK_PGN) \
		--move_column transcript

generate_precompute_random_benchmark_games: generate_benchmark_games precompute_benchmark

double_benchmark_model:
	$(PYTHON) evaluation/$(BENCHMARK1) \
		eval \
		--checkpoint $(CHECKPOINT) \
		--precomputed_moves evaluation/eval_datasets/lichess13_100g_180m.pkl \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \

	$(PYTHON) evaluation/$(BENCHMARK1) \
		eval \
		--checkpoint $(CHECKPOINT) \
		--precomputed_moves evaluation/eval_datasets/random100games.pkl \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \



full_benchmark: generate_precompute_random_benchmark_games benchmark_model


show_dic:
	python evaluation/show_dictionary.py

benchmark_set:
##2K
	$(PYTHON) evaluation/$(BENCHMARK1) \
		eval \
		--checkpoint evaluation/eval_models/random16M_8layer_2K.pth \
		--precomputed_moves evaluation/eval_datasets/lichess13_100g_180m.pkl \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \

##12K
	$(PYTHON) evaluation/$(BENCHMARK1) \
		eval \
		--checkpoint evaluation/eval_models/random16M_8layer_12K.pth \
		--precomputed_moves evaluation/eval_datasets/lichess13_100g_180m.pkl \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \


##23K
	$(PYTHON) evaluation/$(BENCHMARK1) \
		eval \
		--checkpoint evaluation/eval_models/lichess9gb_8layer_23K.pth \
		--precomputed_moves evaluation/eval_datasets/lichess13_100g_180m.pkl \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \


##22K
	$(PYTHON) evaluation/$(BENCHMARK1) \
		eval \
		--checkpoint evaluation/eval_models/random16M_8layer_22K.pth \
		--precomputed_moves evaluation/eval_datasets/lichess13_100g_180m.pkl \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \

	$(PYTHON) evaluation/$(BENCHMARK1) \
		eval \
		--checkpoint evaluation/eval_models/random16M_8layer_22K.pth \
		--precomputed_moves evaluation/eval_datasets/random100games.pkl \
		--data_dir $(DATA_DIR) \
		--results_file $(RESULTS_FILE) \
		--temperature $(TEMPERATURE) \



remote_benchmark_model:
	sky jobs launch -c benchmarkCluster remote/benchmark.yaml


	
hfdataset:
	sky jobs launch -c boardCluster remote/hfExport.yaml

clean_bins:
	rm -f data/lichess_hf_dataset/train.bin
	rm -f data/lichess_hf_dataset/val.bin
	rm -f data/lichess_hf_dataset/train*.bin
	rm -f data/lichess_hf_dataset/val*.bin
	rm -f data/lichess_hf_dataset/*.csv
	rm -f data/lichess_hf_dataset/binned/train*.bin
	rm -f data/lichess_hf_dataset/binned/val*.bin

test:
	python test.py	
sand:
	python sandbox.py


