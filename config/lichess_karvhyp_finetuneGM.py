out_dir = "chess_checkpoints"
eval_interval = 5000 ##this is too large
eval_iters = 100
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 1000  # don't print too too often

always_save_checkpoint = True

wandb_log = True 
mlflow_log = False 
wandb_project = "karvhyp-chess-lichess-finetune-GM"
wandb_run_name = "RUN600"

dataset = "wc"
gradient_accumulation_steps = 1
batch_size = 100
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)
init_from = 'resume'

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-5
max_iters = 100000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 3e-6  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small
grad_clip = 1.0

compile = False

data_type = 'GM'
checkpoint_key_prefix = f"lichess_karvhyp_finetune_GM"
bucket_name = 'chess-gpt-checkpoints-finetune'
data_bucket_name = "bins-bucket-craft"
verbose = True
debugging = False