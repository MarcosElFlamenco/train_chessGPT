"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import zipfile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from download_dataset import download_bins_from_s3

from model import GPTConfig, GPT
from remote.connecting_checkpoints import upload_checkpoint, load_checkpoint, download_bins_from_s3_with_progress

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
mlflow_log = False # disabled by default
mlflow_location = 'mlflow_storage'
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'lichess_hf_dataset'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

data_type = '9gb'
checkpoint_key_prefix = 'lichess_8layer'
bucket_name = 'chess-checkpoint-craft'

verbose = False
local_bypass = False

# system
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
print('config keys:', config_keys)
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
#s3 settings
print(f"decay lr {decay_lr}")

data_bucket_name = "chess-data-bucket-craft"
data_dir = os.path.join('data', dataset)

print(data_type)

train_name = f'train{data_type}.bin'
val_name = f'val{data_type}.bin'
train_path = os.path.join(data_dir, train_name)
val_path = os.path.join(data_dir, val_name)

print(f" the s3 bucket is {data_bucket_name}")
if os.path.isfile(train_path):
    print("Train file already exists, skipping download")
else:
    print(f"Downloading {train_name} to the following adress {train_path}...")
    download_bins_from_s3_with_progress(bucket_name=data_bucket_name, object_name=train_name,file_name=train_path )
if os.path.isfile(val_path):
    print("Val file already exists, skipping download")
else:
    print(f"Downloading {val_name} ... to the following adress {val_path}...")
    download_bins_from_s3_with_progress(bucket_name=data_bucket_name, object_name=val_name,file_name=val_path )
#with zipfile.ZipFile(zipped_train_name, 'r') as zip_ref:
#zip_ref.extractall(train_name)
#with zipfile.ZipFile(zipped_val_name, 'r') as zip_ref:
#zip_ref.extractall(val_name)



# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



##Ill start here
# poor man's data loader
train_data = np.memmap(train_path, dtype=np.uint8, mode='r')
val_data = np.memmap(val_path, dtype=np.uint8, mode='r')

np.set_printoptions(threshold=np.inf)
with open('first_few_lines.txt', 'w') as f:
# Decode and print as a continuous string
    output = ''
    for i in range(4000):
#    output = train_data[1024 * i:1024 * (i+1)].tobytes().decode('utf-8').replace('\n', '')
# Convert to a single-line string
        output += ' '.join(map(str, train_data[1024*i:1024*(i+1)])) + '\n'

    f.write(output)
    print('written')
    exit()



def get_batch(split):
    data = train_data if split == 'train' else val_data
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    # Ensure the starting index is a multiple of block_size
    ix = torch.randint(0, len(data) // (block_size + 1), (batch_size,)) * (block_size + 1)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
train_loss_list = []
val_loss_list = []

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
print(meta_path)
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line