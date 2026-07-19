"""
Training script for Fade (3.1T total / 44.5B active parameter MoE code model).

Same core recipe as before (AdamW, cosine schedule, grad accumulation,
mixed precision, DDP), with one MoE-specific addition: the auxiliary
load-balancing loss is already summed into the loss returned by the model,
so no extra wiring is needed here.

Read the honesty section in README.md before spending real money running
this — the model-parallelism requirements at this scale go beyond what
this single script handles alone (see notes below).
"""

import os
import time
import math
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config import FadeConfig, count_params
from model import Fade
from data import get_dataloader


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--batch_size", type=int, default=1, help="per-GPU micro batch size")
    p.add_argument("--grad_accum_steps", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=500_000)
    p.add_argument("--warmup_steps", type=int, default=3_000)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--min_lr", type=float, default=1.5e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * step / args.warmup_steps
    if step > args.max_steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def main():
    args = get_args()

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_master = True

    torch.manual_seed(1337 + rank)
    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)

    cfg = FadeConfig()
    if is_master:
        total, active, _ = count_params(cfg)
        print(f"{cfg.name}: {total:,} total params ({total/1e12:.2f}T), "
              f"{active:,} active params ({active/1e9:.1f}B)")
        print(
            "NOTE: at this parameter count, the full model does not fit on a "
            "single GPU (or likely a single node). This script assumes an "
            "external model-parallel / expert-parallel wrapper (e.g. DeepSpeed-MoE, "
            "Megatron-Core MoE, or a similar framework) shards `Fade` across many "
            "devices. Running this file as-is only works at small test configs."
        )

    model = Fade(cfg).to(device)
    raw_model = model

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    decay_params = [p for n, p in raw_model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in raw_model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    start_step = 0
    if args.resume:
        ckpt_path = os.path.join(args.out_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            if is_master:
                print(f"Resumed from step {start_step}")

    train_loader = get_dataloader(args.data_dir, args.batch_size, cfg.block_size,
                                   split="train", fim_rate=cfg.fim_rate)

    model.train()
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            x, y = next(train_loader)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)

            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                 dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, loss = model(x, y)
                loss = loss / args.grad_accum_steps

            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()

        if is_master and step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_step = args.batch_size * args.grad_accum_steps * cfg.block_size * world_size
            print(f"step {step} | loss {accumulated_loss:.4f} | lr {lr:.2e} "
                  f"| tokens/sec {tokens_per_step / max(dt, 1e-6):,.0f}")

        if is_master and step % args.save_interval == 0 and step > 0:
            ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": cfg}
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
