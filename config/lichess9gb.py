out_dir = "chess_saver"
eval_interval = 1000 ##this is too large
eval_iters = 100
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 100  # don't print too too often

always_save_checkpoint = True

wandb_log = True 
mlflow_log = False 
wandb_project = "chess-gpt-lichess-train"
wandb_run_name = "vocab32_lichess"

dataset = "lichess_hf_dataset"
gradient_accumulation_steps = 10
batch_size = 20
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)
init_from = 'resume'

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 1e-4
max_iters =  200000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small
grad_clip = 0.0

warmup_iters = 2000  # not super necessary potentially
compile = True 

data_type = '9gb'
checkpoint_key_prefix = f"lichess{data_type}_vocab32"
bucket_name = 'chess-gpt-checkpoints'
data_bucket_name = "bins-bucket-craft"
verbose = False