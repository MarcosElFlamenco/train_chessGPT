out_dir = "chess_saver"

eval_interval = 5
eval_iters = 1
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 1  # don't print too too often

always_save_checkpoint = False

wandb_log = True
mlflow_log = False
mlflow_location = 'arn:aws:sagemaker:us-east-1:278996676838:mlflow-tracking-server/craft-flow'
wandb_project = "chess-gpt-batch"
wandb_run_name = "8layer_lichess"
local_bypass = True

dataset = "lichess_hf_dataset"
gradient_accumulation_steps = 1
batch_size = 1
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)
init_from = 'resume'
data_type = '1M'

# baby GPT model :)
n_layer= 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 0  # not super necessary potentially
compile = False
verbose = False
