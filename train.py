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
dataset = 'openwebtext'
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

data_type = '1M'
checkpoint_key_prefix = 'lichess_8layer'
bucket_name = 'chess-checkpoint-craft'

verbose = False
local_bypass = False
debugging = True

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
data_dir = os.path.join('data')

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


def get_batch(split):
    data = train_data if split == 'train' else val_data
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    # Ensure the starting index is a multiple of block_size
    ix = torch.randint(0, len(data) // (block_size + 1), (batch_size,)) * (block_size + 1)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if debugging: 
        print(f'These are the first few lines of the batch inputs: \n {x[0][0:10]}... \n {x[1][0:10]}... \n {x[2][0:10]}...')
        print(f'These are the first few lines of the batch targets: \n {y[0][0:10]}... \n {y[1][0:10]}... \n {y[2][0:10]}...')
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


if init_from == 'resume':
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = load_checkpoint(bucket_name, checkpoint_key_prefix, device)
    if checkpoint == None:
        print('There was no checkpoint')
        init_from = 'scratch'
    else:
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        print(f"Rebuilding model from checkpoint {checkpoint}")
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
            print(f"{k} : {checkpoint_model_args[k]}")
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        iter_num = checkpoint['iter_num']
        try:
            train_loss_list = checkpoint['train_loss_list']
        except Exception as e:
            print('no train loss list found defaulting to empty')
        try:
            val_loss_list = checkpoint['val_loss_list']
        except Exception as e:
            print('no val loss list found defaulting to empty')
        flag = False
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                flag = True
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        best_val_loss = checkpoint['best_val_loss']
        if flag:
            print('we did in fact remove unwanted prefix')
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of chessGPT to 32")
    else:
        print(f"Taking the meta vocab size of {meta_vocab_size}")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 32
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

print(f"gradient will be clipped at {grad_clip}")
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    #wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.init(project=wandb_project, config=config)
if mlflow_log and master_process:
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(mlflow_location)
#    mlflow.log_params(config)
    mlflow.set_experiment("chess_training")



#with mlflow.start_run(log_system_metrics=True):
## training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
grad_norms = []
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        train_loss_list.append(losses['train'])
        val_loss_list.append(losses['val'])
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss_eval": losses['train'],
                "val/loss_eval": losses['val'],
            })


        if mlflow_log:
            mlflow.log_metric('iter', iter_num ) 
            mlflow.log_metric('train_loss', losses['train'])
            mlflow.log_metric('val_loss', losses['val'])
            mlflow.log_metric('lr', lr)
            mlflow.log_metric('mfu', running_mfu*100)

        if (losses['val'] < best_val_loss or always_save_checkpoint):
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'train_loss_list': train_loss_list,
                    'val_loss_list' : val_loss_list,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                checkpoint_key = checkpoint_key_prefix + f"_{iter_num//1000}" + "K.pth"
                local_file_path = os.path.join(out_dir, checkpoint_key)
                torch.save(checkpoint, local_file_path)
                print(f'upload checkpoint {checkpoint_key} to bucket {bucket_name}')
                if local_bypass:
                    print("upload bypassed because local")
                else:
                    if eval_only and iter_num == 0:
                        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        #if iter_num < 10:
        #     print("Batch")
        #     print(X)
        #     print("y")
        #     print(Y)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if verbose:
            new_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 200)
            print(f"The old gradient norm was {total_norm} now it's {new_norm}")
        grad_norms.append(total_norm)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        grad_avg, grad_min, grad_max = 0,0,0
        if(len(grad_norms) != 0):
            grad_avg = sum(grad_norms)/len(grad_norms)
            grad_min = min(grad_norms)
            grad_max = max(grad_norms)
        grad_norms = []
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "avg_gardient_norm": grad_avg,
                "max_gradient_norm": grad_max,
                "min_gradient_norm": grad_min
            })


        if mlflow_log:
            mlflow.log_metric('iter', iter_num ) 
            mlflow.log_metric('train_loss', lossf)
            mlflow.log_metric('lr', lr)
            mlflow.log_metric('mfu', running_mfu*100)



    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
