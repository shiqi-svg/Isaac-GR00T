"""
Benchmark script for GR00T-N1.6 action inference timing and TFLOPS estimation.

Usage:
    cd /home/user/Isaac-GR00T
    uv run python scripts/benchmark_inference.py \
      --model-path nvidia/GR00T-N1.6-3B \
      --dataset-path demo_data/gr1.PickNPlace \
      --embodiment-tag GR1 \
      --video-backend ffmpeg \
      --warmup-iters 3 \
      --benchmark-iters 10
"""

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops_per_inference(model_config, num_denoising_steps=4):
    """
    Estimate FLOPs for a single action inference.

    Components:
    1. Eagle Backbone forward pass (vision + language)
    2. DiT denoising loop (num_denoising_steps iterations)
       - State encoding
       - Action encoding
       - DiT forward (32 layers of cross/self attention + FFN)
       - Action decoding
    """
    # ---- Backbone (Eagle) ----
    # ~2B params, ~2 FLOPs per param per forward (MatMul dominant)
    backbone_params = 2_000_000_000
    backbone_flops = 2 * backbone_params

    # ---- DiT Action Head (per denoising step) ----
    dit_params = getattr(model_config, 'dit_total_params', 1_091_722_240)

    # Each DiT forward: ~2 * params FLOPs (transformer attention + FFN)
    # Sequence length: 1 (state) + 16 (actions) = 17 tokens
    # Cross-attention to VL embeddings (~256 tokens): adds ~2 * seq_q * seq_kv * dim
    seq_len = 17
    vl_seq_len = 256  # approximate VL embedding sequence length
    num_layers = 32
    hidden_dim = 1536
    num_heads = 32
    head_dim = 48

    # Self-attention FLOPs per layer: 4 * seq_len^2 * hidden_dim (Q,K,V projections + attention)
    self_attn_flops = 4 * seq_len * seq_len * hidden_dim + 2 * seq_len * hidden_dim * hidden_dim
    # Cross-attention FLOPs per layer: 2 * seq_len * vl_seq_len * hidden_dim + QKV projections
    cross_attn_flops = 4 * seq_len * vl_seq_len * hidden_dim + 2 * seq_len * hidden_dim * hidden_dim
    # FFN FLOPs per layer: 2 * seq_len * hidden_dim * 4 * hidden_dim (typically 4x expansion)
    ffn_flops = 2 * seq_len * hidden_dim * 4 * hidden_dim

    # Total per layer (alternating self/cross attention)
    per_layer_flops = (self_attn_flops + cross_attn_flops) / 2 + ffn_flops
    dit_forward_flops = num_layers * per_layer_flops

    # Action encoder + decoder FLOPs (small MLPs)
    encoder_decoder_flops = 2 * seq_len * hidden_dim * 1024 * 4  # approximate

    # Total per denoising step
    per_step_flops = dit_forward_flops + encoder_decoder_flops

    # Total inference FLOPs
    total_flops = backbone_flops + num_denoising_steps * per_step_flops

    return {
        "backbone_flops": backbone_flops,
        "dit_forward_flops_per_step": dit_forward_flops,
        "encoder_decoder_flops_per_step": encoder_decoder_flops,
        "per_denoising_step_flops": per_step_flops,
        "total_flops": total_flops,
        "num_denoising_steps": num_denoising_steps,
    }


def parse_observation_gr00t(obs, modality_configs):
    """Parse observation into the format expected by Gr00tPolicy.
    Must group by modality: {"video": {...}, "state": {...}, "language": {...}}
    """
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def main():
    parser = argparse.ArgumentParser(description="Benchmark GR00T-N1.6 inference")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--embodiment-tag", type=str, default="GR1")
    parser.add_argument("--video-backend", type=str, default="ffmpeg")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=16)
    args = parser.parse_args()

    embodiment_tag = EmbodimentTag(args.embodiment_tag.lower())

    print("=" * 80)
    print("GR00T-N1.6 Inference Benchmark")
    print("=" * 80)

    # ---- Step 1: Load model ----
    print("\n[1/5] Loading model...")
    t0 = time.time()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=embodiment_tag,
        device="cuda:0",
    )
    model_load_time = time.time() - t0
    print(f"  Model loaded in {model_load_time:.2f}s")

    # ---- Step 2: Count parameters ----
    print("\n[2/5] Counting parameters...")
    model = policy.model

    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters:     {total_params:>15,}")
    print(f"  Trainable parameters: {trainable_params:>15,}")

    # Count backbone vs action head separately
    if hasattr(model, 'backbone'):
        bb_total, bb_train = count_parameters(model.backbone)
        print(f"  Backbone parameters:  {bb_total:>15,} (trainable: {bb_train:,})")
    if hasattr(model, 'action_head'):
        ah_total, ah_train = count_parameters(model.action_head)
        print(f"  Action Head params:   {ah_total:>15,} (trainable: {ah_train:,})")

    # ---- Step 3: Estimate FLOPs ----
    print("\n[3/5] Estimating FLOPs...")
    flops_info = estimate_flops_per_inference(model.config, args.denoising_steps)
    total_gflops = flops_info["total_flops"] / 1e9
    print(f"  Backbone FLOPs:               {flops_info['backbone_flops']/1e9:.2f} GFLOPs")
    print(f"  DiT forward FLOPs/step:       {flops_info['dit_forward_flops_per_step']/1e9:.2f} GFLOPs")
    print(f"  Total per denoising step:     {flops_info['per_denoising_step_flops']/1e9:.2f} GFLOPs")
    print(f"  Denoising steps:              {flops_info['num_denoising_steps']}")
    print(f"  Estimated total FLOPs:        {total_gflops:.2f} GFLOPs")

    # ---- Step 4: Prepare test data ----
    print("\n[4/5] Preparing test observation...")

    # Load one trajectory to get a real observation
    modality_configs = policy.modality_configs
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend=args.video_backend,
    )

    traj = dataset[0]
    modality_no_action = deepcopy(modality_configs)
    modality_no_action.pop("action")

    data_point = extract_step_data(traj, 0, modality_no_action, embodiment_tag)

    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for lang_key in modality_configs["language"].modality_keys:
        obs[lang_key] = data_point.text

    parsed_obs = parse_observation_gr00t(obs, modality_configs)
    print(f"  Observation keys: {list(parsed_obs.keys())}")

    # ---- Step 5: Benchmark inference ----
    print(f"\n[5/5] Running benchmark ({args.warmup_iters} warmup + {args.benchmark_iters} timed iterations)...")

    # Warmup
    print("  Warming up...")
    for i in range(args.warmup_iters):
        with torch.inference_mode():
            _ = policy.get_action(parsed_obs)
        torch.cuda.synchronize()
        print(f"    Warmup {i+1}/{args.warmup_iters} done")

    # Benchmark: end-to-end (preprocessing + inference)
    e2e_times = []
    inference_only_times = []

    print("  Benchmarking...")
    for i in range(args.benchmark_iters):
        torch.cuda.synchronize()

        # End-to-end timing
        t_start = time.perf_counter()
        with torch.inference_mode():
            action, _ = policy.get_action(parsed_obs)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        e2e_times.append(t_end - t_start)

        print(f"    Iter {i+1}/{args.benchmark_iters}: {(t_end - t_start)*1000:.2f} ms")

    # Benchmark: model.get_action only (skip preprocessing)
    print("\n  Benchmarking model-only (no preprocessing)...")
    # Prepare pre-processed input once
    unbatched = policy._unbatch_observation(parsed_obs)
    processed_inputs = []
    for o in unbatched:
        vla_step = policy._to_vla_step_data(o)
        from gr00t.data.types import MessageType
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
        processed_inputs.append(policy.processor(messages))
    collated = policy.collate_fn(processed_inputs)
    from gr00t.policy.gr00t_policy import _rec_to_dtype
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)

    for i in range(args.warmup_iters):
        with torch.inference_mode():
            _ = policy.model.get_action(**collated)
        torch.cuda.synchronize()

    model_only_times = []
    for i in range(args.benchmark_iters):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.inference_mode():
            _ = policy.model.get_action(**collated)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        model_only_times.append(t_end - t_start)
        print(f"    Iter {i+1}/{args.benchmark_iters}: {(t_end - t_start)*1000:.2f} ms")

    # Benchmark: backbone + action head separately
    print("\n  Benchmarking backbone + action head separately...")
    with torch.inference_mode():
        # model.get_action(**collated) unpacks 'inputs' key, so we need to do the same
        backbone_inputs, action_inputs = policy.model.prepare_input(**collated)
        backbone_out = policy.model.backbone(backbone_inputs)
    torch.cuda.synchronize()

    # Backbone timing
    for i in range(args.warmup_iters):
        with torch.inference_mode():
            _ = policy.model.backbone(backbone_inputs)
        torch.cuda.synchronize()

    backbone_times = []
    for i in range(args.benchmark_iters):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.inference_mode():
            _ = policy.model.backbone(backbone_inputs)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        backbone_times.append(t_end - t_start)

    # Action head (DiT denoising) timing
    print("  Benchmarking action head (DiT denoising) only...")
    torch.cuda.synchronize()

    for i in range(args.warmup_iters):
        with torch.inference_mode():
            _ = policy.model.action_head.get_action(backbone_out, action_inputs)
        torch.cuda.synchronize()

    dit_times = []
    for i in range(args.benchmark_iters):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.inference_mode():
            _ = policy.model.action_head.get_action(backbone_out, action_inputs)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        dit_times.append(t_end - t_start)

    # ---- Results ----
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    def print_stats(name, times_ms):
        arr = np.array(times_ms) * 1000  # to ms
        print(f"\n  {name}:")
        print(f"    Mean:   {arr.mean():.2f} ms")
        print(f"    Std:    {arr.std():.2f} ms")
        print(f"    Min:    {arr.min():.2f} ms")
        print(f"    Max:    {arr.max():.2f} ms")
        print(f"    P50:    {np.percentile(arr, 50):.2f} ms")
        print(f"    P90:    {np.percentile(arr, 90):.2f} ms")
        print(f"    P99:    {np.percentile(arr, 99):.2f} ms")
        return arr.mean()

    mean_e2e = print_stats("End-to-End (preprocess + inference)", e2e_times)
    mean_model = print_stats("Model Only (backbone + action head)", model_only_times)
    mean_backbone = print_stats("Backbone Only (Eagle VLM)", backbone_times)
    mean_dit = print_stats("Action Head Only (DiT denoising x{})".format(args.denoising_steps), dit_times)

    print(f"\n  Preprocessing overhead: ~{mean_e2e - mean_model:.2f} ms")

    # ---- TFLOPS Calculation ----
    print("\n" + "=" * 80)
    print("TFLOPS ESTIMATION")
    print("=" * 80)

    # Method 1: Based on FLOPs estimate
    tflops_estimate = total_gflops / (mean_model / 1000) / 1000  # GFLOPs / seconds / 1000 = TFLOPS
    print(f"\n  Method 1: Analytical FLOPs estimate")
    print(f"    Estimated FLOPs per inference: {total_gflops:.2f} GFLOPs")
    print(f"    Mean model latency:           {mean_model:.2f} ms")
    print(f"    Estimated throughput:          {tflops_estimate:.2f} TFLOPS")

    # Method 2: Parameter-based estimate (2 * params per forward * denoising_steps for DiT)
    backbone_flops_approx = 2 * (bb_total if hasattr(model, 'backbone') else 2e9)
    dit_flops_approx = 2 * (ah_total if hasattr(model, 'action_head') else 1e9) * args.denoising_steps
    total_flops_param = (backbone_flops_approx + dit_flops_approx) / 1e9  # GFLOPs
    tflops_param = total_flops_param / (mean_model / 1000) / 1000
    print(f"\n  Method 2: Parameter-count based (2*params per forward)")
    print(f"    Backbone: ~{backbone_flops_approx/1e9:.2f} GFLOPs")
    print(f"    DiT head: ~{dit_flops_approx/1e9:.2f} GFLOPs ({args.denoising_steps} steps)")
    print(f"    Total:    ~{total_flops_param:.2f} GFLOPs")
    print(f"    Estimated throughput:          {tflops_param:.2f} TFLOPS")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    freq = 1000.0 / mean_e2e
    print(f"  Model:                  GR00T-N1.6-3B")
    print(f"  Total params:           {total_params/1e9:.2f}B")
    print(f"  Denoising steps:        {args.denoising_steps}")
    print(f"  Action horizon:         {args.action_horizon}")
    print(f"  End-to-end latency:     {mean_e2e:.2f} ms")
    print(f"  Model-only latency:     {mean_model:.2f} ms")
    print(f"    - Backbone:           {mean_backbone:.2f} ms")
    print(f"    - DiT action head:    {mean_dit:.2f} ms")
    print(f"  Inference frequency:    {freq:.1f} Hz")
    print(f"  Est. TFLOPS:            {tflops_estimate:.2f} (analytical) / {tflops_param:.2f} (param-based)")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_mem_used = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\n  GPU:                    {gpu_name}")
        print(f"  GPU Memory Total:       {gpu_mem_total:.1f} GB")
        print(f"  GPU Memory Peak Used:   {gpu_mem_used:.1f} GB")

    print("=" * 80)


if __name__ == "__main__":
    main()
