PYTHON := python3
TRAIN := train.py
PREPARE := data/lichess_hf_dataset/prepare.py

run train: $(TRAIN) 
	$(PYTHON) $(TRAIN) \
		config/local.py

prepare: $(PREPARE)
	$(PYTHON) $(PREPARE)

remote: $(MAIN) clean
	aws s3 rm s3://go-bucket-craft --recursive
	sky jobs launch -c boardCluster --env WANDB_API_KEY remote/lichess.yaml

toyremote: $(MAIN) clean
	aws s3 rm s3://go-bucket-craft --recursive
	sky jobs launch -c boardCluster --env WANDB_API_KEY remote/toy.yaml


hfdataset:
	sky jobs launch -c boardCluster remote/hfExport.yaml

clean:
	rm -f data/lichess_hf_dataset/train.bin
	rm -f data/lichess_hf_dataset/val.bin
