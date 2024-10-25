PYTHON := python3
TRAIN := train.py
PREPARE := data/lichess_hf_dataset/prepare.py
CHECKPOINT := checkpointBeta.pth
INPUT_PGN := ';1. c3 c6 2. e3 Nf6 3. Ba6 Nd5 4. Bd3 f5 5. f4 Nxe3 6. b4 e6 7. c4 Nc2+ 8. Kf1 Bxb4 9. Nh3 Nxa1 10. Bb2 b6 11. Ng5 Bf8 12. Be5 Ba6 13. Qh5+ Ke7 14. Bc3 Bb7 15. Kg1 e5 16. Bc2 exf4 17. Qg6 Nb3 18. Na3 f3 19. Qe6+ dxe6 20. Bf6+ Kd7 21. Be7 Ba6 22. Bb1 Na5 23. Nxf3 Bxe7 24. Ne1 Kc7 25. Bxf5 Qxd2 26. Bh3 Qd5 27. g3 Bb4 28. Bg2 Bxa3 29. g4 h6 30. Bxd5 cxd5 31. Nf3 Rf8 32. h4 b5 33. g5 Be7 34. gxh6 Kd6 35. Rh3 Rh8 36. Ne1 Kc7 37. Rg3 Rxh6 38. Ra3 Bxa3 39. Kf1 Kb6 40. Nc2 bxc4 41. Ne3 Bb4 42. Nxc4+ Bxc4+ 43. Kg2 Be7 44. Kh3 Ba6 45. Kh2 Rh5 46. Kg3 Ba3 47. Kg4 Kb7 48. Kh3 Be7 49. Kh2 d4 50. a3 Rh6 51. Kh3 g5 52. a4 d3 53. Kg3 Bc5 54. Kh2 Bd6+ 55. Kh1 Bh2 56. Kg2 d2 57. Kh3 gxh4 58. Kg4 Nc4 59. Kg5 Bb5 60. Kxh6 Ba6 61. Kg6 Nd7 62. Kh7 Bc7 63. Kg7 Kb6 64. Kf7 Nd6+ 65. Kxe6 Nc8 66. Kxd7 d1=R+ 67. Ke6 Ne7 68. Kf6 Rf8+ 69. Kxe7 Bd3 70. Kd7 Rd2 71. a5+ Ka6 72. Ke7 Rh2 73. Kd7 Rh3 74. Ke6 Bxa5 75. Ke7 Kb7 76. Kd7 Re8 77. Kxe8 Bc3 78. Ke7 Rh1 79. Kd8 Ba1 80. Ke7 Rd1 81. Kd7 Rh1 82. Ke7 Bb2 83. Kf'
##INPUT_PGN := ";1. Nf3 d6 2. c4 Qd7 3. h4 a6 4. e4 Qg4 5. b3 Qxe4+ 6. Be2 g5 7. g3 Qxh4 8. a3 Ra7 9. O-O Qe4 10. c5 f5 11. Re1 Qxb1 12. Bc4 Qb2 13. Bd3 Qb1 14. Nh2 Kd8 15. Re5 Kd7 16. Rxb1 h5 17. f4 Nc6 18. Rb2 Kd8 19. Rb1 b5 20. fxg5 Rh7 21. Nf3 b4 22. axb4 Be6 23. Rxf5 Rg7 24. Be2 Ne5 25. Kf1 Bd5 26. Rf4 h4 27. Bb2 Nd7 28. Rxf8+ Nxf8 29. Kg2 e5 30. d3 Kd7 31. Ra1 Ra8 32. Ra3 Bxb3 33. Kh3 a5 34. b5 Nh7 35. Qe1 a4 36. Nh2 Ra6 37. b6 Rxg5 38. Bd4 Kc6 39. Qf1 Ba2 40. Qf8 Rxg3+ 41. Kxh4 Rxd3 42. Qf5 Be6 43. Qg4 Nh6 44. Qg8 Rh3+ 45. Rxh3 Bb3 46. Bc4 Rxb6 47. Rd3 Nf5+ 48. Kg4 Rb4 49. Qg6 Nh4 50. Rg3 Ng5 51. Qh5 Nh7 52. Bg8 Ng6 53. Qh3 Nf6+ 54. Kg5 Kb5 55. Qf5 Nd7 56. Qc2 Ndf8 57. Qd1 Nh7+ 58. Kf5 Rc4 59. Qg4 Bc2+ 60. Qe4 dxc5 61. Rg2 c6 62. Qxc2 Nf4 63. Ba1 Nxg2 64. Qd2 Re4 65. Bb3 Rb4 66. Ba2 Nf6 67. Qd1 a3 68. Qb3 Nh4+ 69. Ke6 Ne8 70. Qc4+ Kb6 71. Qb5+ Kxb5 72. Nf3 Rb1 73. Bd5 Nc7+ 74. Kd6 Rc1 75. Bb2 e4 76. Be6 Ng6 77. Bg4 Ne5 78. Bxa3 Ra1 79. Nxe5 Ra2 80. Bd1 Na6 81. Bg4 Rc2 82. Be6 Ka5 83. Bb2 Rxb2 84. Bc8 Re2 85"
##INPUT_PGN := ";1. Nf3 d6 2. c4 Qd7 3. h4 a6 4. e4 Qg4 5. b3 Qxe4+ 6. Be2 g5 7. g3 Qxh4 8. a3 Ra7 9. O-O Qe4 10. c5 f5 11. Re1 Qxb1 12. Bc4 Qb2"
##INPUT_PGN := ';1. d4 Nf6 2. Nf3 d5 3. e3 c5 4. Nbd2 cxd4 5. exd4 Qc7 6. c3 Bd7 7. Bd3 Nc6 8. O-O Bg4 9. Re1 e6 10. Nf1 Bd6 11. Bg5 O-O 12. Bxf6 gxf6 13. Ng3 f5 14. h3 Bxf3 15. Qxf3 Ne7 16. Nh5 Kh8 17. g4 Rg8 18. Kh1 Ng6 19. Bc2 Nh4 20. Qe3 Rg6 21. Rg1 f4 22. Qd3 Qe7 23. Rae1 Qg5 24. c4 dxc4 25. Qc3 b5 26. a4 b4 27. Qxc4 Rag8 28. Qc6 Bb8 29. Qb7 Rh6 30. Be4 Rf8 31. Qxb4 Qd8 32. Qc3 Ng6 33. Bg2 Qh4 34. Re2 f5 35. Rxe6 Rxh5 36. gxh5 Qxh5 37. d5+ Kg8 38. d6'
##INPUT_PGN := ';1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd2 d5 6. a3 Be7 7. Nf3 c5 8. dxc5 Bxc5 9. Qc2 dxc4 10. Bxc4 Nbd7 11. Rd1 Be7 12. Ng5 h6 13. h4 Qc7 14. Be2 Rd8 15. Rc1 Nf8 16. Nge4 Nxe4 17. Nxe4 Qxc2 18. Rxc2 Bd7 19. Bb4 Bxb4+ 20. axb4 Bc6 21. Nc5 Bxg2 22. Rg1 Bd5 23. e4 Bc6 24. b5 Be8 25. Nxb7 Rd4 26. Rc4 Rd7 27. Nc5 Rc7 28. Rc3 Rac8 29. b4 Nd7 30. Rcg3 Nxc5 31. bxc5 Rxc5 32. Rxg7+ Kf8 33. Bd3 Rd8 34. Ke2 Rc3 35. Rg8+ Ke7 36. R1g3 e5 37. Rh8 Rd6 38. b6 Rxb6 39. Rxe8+ Kxe8 40. Bb5+ Rxb5 41. Rxc3 Kd7 42. Rf3 Ke7 43. Rc3 a5 44. Rc7+ Kf6 45. Rc6+ Kg7 46. Ra6 Rb2+ 47. Kf3 Ra2 48. Kg3 h5 49. Ra8 Ra1 50. Kg2 a4 51. Ra5 f6 52. Kf3 a3 53. Ra6 Kf7 54. Ke3 Ke8 55. Ke2 Ke7 56. Kf3 Ra2 57. Ke3 Ra1 58. Ke2 Kf7 59. Kf3 Ra2 60. Ke3 Ke7 61. Kf3 Kd7 62. Rxf6 Rb2 63. Ra6 Rb3+ 64. Kg2 Kc7 65. f4 exf4 66. e5 Kb7 67. Ra4 Kc6 68. Ra6+ Kb5 69. Ra7 Kb6 70. Ra8 Kc5 71. Ra6 Kb5 72. Ra7 Kb6 73. Ra8 Kc6 74. Ra6+ Kd7 75. Kf2 Ke7 76. Kg2 Re3 77. Kf2 Rg3 78. Kf1 Rc3 79. Kf2 Re3 80. Kg2 Kd7 81. Kf2 Kc7 82. e6 Kd8'
DATA_DIR := data/lichess_hf_dataset
TEMPERATURE := 1.0


run train: $(TRAIN) 
	$(PYTHON) $(TRAIN) \
		config/local.py

prepare: $(PREPARE)
	$(PYTHON) $(PREPARE)

remote_train: clean_bins
	sky jobs launch -c boardCluster --env WANDB_API_KEY remote/lichess.yaml

toyremote: $(MAIN)
	sky jobs launch -c boardCluster --env WANDB_API_KEY remote/toy.yaml

generate:
	$(PYTHON) model_generate.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR)

gen_test:
	$(PYTHON) gen_tester.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR) \
		--temperature $(TEMPERATURE)


gen_and_val:
	$(PYTHON) gen_and_val.py \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT_PGN) \
		--data_dir $(DATA_DIR) \
		--temperature $(TEMPERATURE) \
		--graph

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


