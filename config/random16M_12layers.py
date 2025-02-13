out_dir = "chess_saver"
eval_interval = 2000 ##this is too large
eval_iters = 100
# I'm not sure what's going on, but when log_interval == 100, the time per iter is inaccurate and much longer than it should be
# when running on multiple GPUs. TODO: investigate
log_interval = 100  # don't print too too often

always_save_checkpoint = False

wandb_log = True 
mlflow_log = False 
wandb_project = "chess-gpt-lichesstrain"
wandb_run_name = "8layer_lichess"

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

learning_rate = 3e-4
max_iters = 50000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000  # not super necessary potentially
compile = False

<<<<<<< HEAD:config/random16M_12layers.py
data_type = '16MnoS'
checkpoint_key = 'random16M_8layer_A.pth'
bucket_name = 'chess-gpt-checkpoints'
verbose = True
=======
data_type = '9gb'
checkpoint_key = 'lichess_8layer_16M.pth'
bucket_name = 'bigger-model'
>>>>>>> d0ee931 (local is right config):config/lichess16M.py
