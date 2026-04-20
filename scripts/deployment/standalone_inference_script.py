from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Literal
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Combined inference script supporting both PyTorch and TensorRT modes.

Example commands:
 
# PyTorch mode (default):
python groot/scripts/deployment/standalone_inference_script.py \
  --model_path /path/to/checkpoint \
  --dataset_path /path/to/dataset \
  --embodiment_tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode pytorch

# TensorRT mode:
python groot/scripts/deployment/standalone_inference_script.py \
  --model_path /path/to/checkpoint \
  --dataset_path /path/to/dataset \
  --embodiment_tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode tensorrt \
  --trt_engine_path ./groot_n1d6_onnx/dit_model_bf16.trt
"""

###############################################################################
# TENSORRT Module Wrappers
###############################################################################


def set_seed(seed: int = 0):
    """
    Set seed for all random number generators.
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA ops
    torch.use_deterministic_algorithms(True, warn_only=True)

    # For cuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch requires this to be set for some CUDA kernels
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TensorRTDiTWrapper:
    """Wrapper for TensorRT DiT engine."""

    def __init__(self, engine_path: str, device: int = 0):
        import tensorrt as trt

        self.device = device

        # Ensures CUDA driver is properly loaded
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(device)  # Set the specified CUDA device
            logging.info(f"CUDA initialized via PyTorch: device {device}")
        else:
            raise RuntimeError("CUDA not available for TensorRT")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        logging.info(f"TensorRT engine loaded: {engine_path}")

    def __call__(self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None):
        """Forward pass through TensorRT DiT."""
        # Setup context bindings
        sa_embs = sa_embs.to(f"cuda:{self.device}").contiguous()
        vl_embs = vl_embs.to(f"cuda:{self.device}").contiguous()
        timestep = timestep.to(f"cuda:{self.device}").contiguous()  # Keep as int64

        if image_mask is not None:
            image_mask = image_mask.to(f"cuda:{self.device}").contiguous()
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(f"cuda:{self.device}").contiguous()

        self.context.set_input_shape("sa_embs", sa_embs.shape)
        self.context.set_input_shape("vl_embs", vl_embs.shape)
        self.context.set_input_shape("timestep", timestep.shape)
        if image_mask is not None:
            self.context.set_input_shape("image_mask", image_mask.shape)
        if backbone_attention_mask is not None:
            self.context.set_input_shape("backbone_attention_mask", backbone_attention_mask.shape)

        self.context.set_tensor_address("sa_embs", sa_embs.data_ptr())
        self.context.set_tensor_address("vl_embs", vl_embs.data_ptr())
        self.context.set_tensor_address("timestep", timestep.data_ptr())
        if image_mask is not None:
            self.context.set_tensor_address("image_mask", image_mask.data_ptr())
        if backbone_attention_mask is not None:
            self.context.set_tensor_address(
                "backbone_attention_mask", backbone_attention_mask.data_ptr()
            )

        # Output in BF16 (matches ONNX export and engine precision)
        output_shape = self.context.get_tensor_shape("output")
        output = torch.empty(
            tuple(output_shape), dtype=torch.bfloat16, device=f"cuda:{self.device}"
        )
        self.context.set_tensor_address("output", output.data_ptr())

        success = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not success:
            raise RuntimeError("TensorRT inference failed")

        return output


def replace_dit_with_tensorrt(policy: Gr00tPolicy | Any, trt_engine_path: str, device: int = 0):
    """Replace the DiT forward method with TensorRT inference."""
    trt_dit = TensorRTDiTWrapper(trt_engine_path, device=device)

    def trt_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        """
        TensorRT wrapper matching DiT forward signature.

        Maps DiT parameter names to ONNX export names:
        - hidden_states -> sa_embs
        - encoder_hidden_states -> vl_embs
        - timestep -> timestep
        - image_mask, backbone_attention_mask passed through
        """
        output = trt_dit(
            sa_embs=hidden_states,
            vl_embs=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

        # DiT returns (output, all_hidden_states) when return_all_hidden_states=True
        if return_all_hidden_states:
            # TensorRT only returns the final output, not intermediate states
            # For inference, we don't need intermediate states, so raise
            # as this seems invalid config for inference
            raise RuntimeError("TensorRT only returns the final output. Check inference config")
        else:
            return output

    policy.model.action_head.model.forward = trt_forward
    logging.info(" DiT replaced with TensorRT engine")


###############################################################################
# TENSORRT Module Wrappers End
###############################################################################


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def prepare_observation_data(
    traj: pd.DataFrame,
    step_count: int,
    modality_configs: dict[str, Any],
    embodiment_tag: EmbodimentTag,
    loader: LeRobotEpisodeLoader,
) -> dict[str, Any]:
    """
    Prepare observation data for inference (CPU-only operations).

    This function is designed to run asynchronously on CPU while GPU performs inference.

    Args:
        traj: Trajectory data
        step_count: Current step in trajectory
        modality_configs: Modality configuration
        embodiment_tag: Embodiment tag
        loader: Data loader with modality configs

    Returns:
        Parsed observation ready for inference
    """
    # Extract step data from trajectory
    data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)

    # Build observation dictionary
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v  # (T, D)
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
    for language_key in loader.modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text

    # Parse observation to expected format
    parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)

    return parsed_obs


def run_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    steps=300,
    action_horizon=16,
    skip_timing_steps=1,
):
    """
    Run inference on a single trajectory.

    Args:
        skip_timing_steps: Number of initial inference steps to skip when calculating timing statistics

    Returns: tuple: (
        state_keys,
        action_keys,
        pred_action_across_time,
        traj,
        actual_steps,
        timing_dict,
    )
    """
    logging.info("\n" + "=" * 80)
    logging.info(f"=== Running Trajectory {traj_id} ===")
    logging.info("=" * 80)

    # Timing accumulators
    timing_dict = {
        "episode_load_time": 0.0,
        "data_prep_times": [],
        "inference_times": [],
    }

    # Load episode
    episode_load_start = time.time()
    traj = loader[traj_id]
    timing_dict["episode_load_time"] = time.time() - episode_load_start

    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    # Inference loop with async prefetching
    num_inference_steps = len(range(0, actual_steps, action_horizon))
    logging.info(f"\nRunning {num_inference_steps} inference steps...")
    logging.info(f"(Skipping first {skip_timing_steps} step(s) for timing statistics)")
    logging.info("Using async prefetching: preparing step i+1 while GPU processes step i")
    logging.info("-" * 80)

    # Create thread pool for async data preparation (single worker is sufficient)
    executor = ThreadPoolExecutor(max_workers=1)

    # List of step counts to process
    step_counts = list(range(0, actual_steps, action_horizon))

    # Prefetch first observation
    future_obs = executor.submit(
        prepare_observation_data,
        traj,
        step_counts[0],
        modality_configs,
        embodiment_tag,
        loader,
    )

    for step_idx, step_count in enumerate(step_counts):
        logging.info(
            f"\n[Step {step_idx + 1}/{num_inference_steps}] Processing timestep {step_count}"
        )

        # Wait for data preparation to complete (should be ready from prefetch)
        data_prep_start = time.time()
        parsed_obs = future_obs.result()  # Blocks until ready
        data_prep_time = time.time() - data_prep_start

        # Prefetch NEXT observation while GPU runs inference on current one
        if step_idx + 1 < len(step_counts):
            next_step_count = step_counts[step_idx + 1]
            future_obs = executor.submit(
                prepare_observation_data,
                traj,
                next_step_count,
                modality_configs,
                embodiment_tag,
                loader,
            )

        # ======== 模型推理（核心入口） ========
        # 在 GPU 上执行推理，同时 CPU 线程在后台准备下一步的数据
        # policy.get_action() 内部完成整个 pipeline：
        #   1. 预处理：图像 resize/normalize，文本 tokenize
        #   2. Eagle Backbone：图像+文本 → 2048维 VL 嵌入（~22ms）
        #   3. State Encoder：29维状态 → 1536维特征
        #   4. DiT 4步去噪：随机噪声 → 16步动作预测（~52ms）
        #   5. 反归一化：模型输出 → 物理关节角度
        # 返回 _action_chunk: dict, 每个key对应一个身体部位，值为 [16, D] 数组
        #   例如: {"left_arm": [16,6], "right_arm": [16,6], "left_hand": [16,6],
        #          "right_hand": [16,6], "waist": [16,5]}
        inference_start = time.time()
        _action_chunk, _ = policy.get_action(parsed_obs)
        inference_time = time.time() - inference_start

        # ======== 计时统计 ========
        # 只记录最后一次推理的耗时（前面的步骤包含 warmup / torch.compile 开销）
        if step_idx == len(step_counts) - 1:
            timing_dict["data_prep_times"].append(data_prep_time)
            timing_dict["inference_times"].append(inference_time)

        # ======== 动作后处理：格式转换 + 拼接为时间序列 ========
        # parse_action_gr00t: 去掉 batch 维度，加上 "action." 前缀
        # 转换后: {"action.left_arm": [16,6], "action.right_arm": [16,6], ...}
        action_chunk = parse_action_gr00t(_action_chunk)

        # 遍历 action_horizon 步（例如=8），将每一步的所有身体部位动作
        # 拼接成一个 29 维向量（6+6+6+6+5=29），追加到 pred_action_across_time 列表
        for j in range(action_horizon):
            # np.atleast_1d 确保单值也变成 1D 数组，避免 concat 时维度不匹配
            # 按 action_keys 顺序拼接: left_arm[j] + right_arm[j] + ... + waist[j]
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            # 追加到时间序列列表，最终形成 [total_steps, 29] 的完整预测轨迹
            pred_action_across_time.append(concat_pred_action)

    # ======== 清理线程池 ========
    executor.shutdown(wait=True)

    logging.info("\n" + "-" * 80)
    logging.info(f"All inference steps completed for current trajectory-id {traj_id}")

    # ======== 收集最后一步的观测数据用于返回 ========
    # 将 parsed_obs 中所有数值类型的数据 flatten 为一维数组并拼接
    # 这部分仅用于结果展示/保存，不影响推理逻辑
    obs = []
    for key in parsed_obs.keys():
        vals = []
        if isinstance(parsed_obs[key], np.ndarray):
            vals.append(parsed_obs[key])
        elif isinstance(parsed_obs[key], list):
            vals.append(np.array(parsed_obs[key]))
        elif isinstance(parsed_obs[key], dict):
            for k in parsed_obs[key].keys():
                vals.append(np.array(parsed_obs[key][k]))

        for val in vals:
            if np.issubdtype(val.dtype, np.number):
                obs.append(val.flatten())
    obs = np.concatenate(obs, axis=-1)

    return (
        state_keys,
        action_keys,
        np.array(pred_action_across_time),
        traj,
        actual_steps,
        timing_dict,
        obs,
    )


def evaluate_predictions(
    state_keys,
    action_keys,
    pred_action_across_time,
    traj,
    traj_id,
    actual_steps,
    action_horizon,
    save_plot_path=None,
):
    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))

    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/stand_alone_inference/traj_{traj_id}.jpeg",
    )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec", "ffmpeg"] = "ffmpeg"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av, ffmpeg: subprocess"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.GR1
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """Inference mode: 'pytorch' (default) or 'tensorrt'."""

    trt_engine_path: str = "./groot_n1d6_onnx/dit_model_bf16.trt"
    """Path to TensorRT engine file (.trt). Used only when inference_mode='tensorrt'."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    skip_timing_steps: int = 3
    """Number of initial inference steps to skip when calculating timing statistics (default: 1 to exclude warmup)."""

    get_performance_stats: bool = True
    """Agreegate and summarize timing and accuracy stats across several runs"""

    seed: int = 42
    """Seed to use for reproducibility."""

    model_action_horizon: int | None = None
    """Override model's internal action_horizon (default: use model config, typically 16). WARNING: values > 16 use untrained position embeddings."""


def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logging.info("\n" + "=" * 80)
    logging.info("=" * 80)
    logging.info(f"Model Path: {args.model_path}")
    logging.info(f"Dataset Path: {args.dataset_path}")
    logging.info(f"Embodiment Tag: {args.embodiment_tag}")
    logging.info(f"Trajectories: {args.traj_ids}")
    logging.info(f"Steps per trajectory: {args.steps}")
    logging.info(f"Action Horizon: {args.action_horizon}")
    logging.info(f"Skip Timing Steps: {args.skip_timing_steps}")
    logging.info(f"Inference Mode: {args.inference_mode}")
    if args.inference_mode == "tensorrt":
        logging.info(f"TensorRT Engine: {args.trt_engine_path}")
    logging.info(f"Seed: {args.seed}")
    set_seed(args.seed)
    logging.info("=" * 80)

    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This script requires a GPU. Exiting.")
        sys.exit(1)

    # Download model checkpoint
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    assert local_model_path is not None, "Provide valid model_path for inference"
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    # Model loading
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 1: Loading Policy ===")
    logging.info("=" * 80)
    model_load_start = time.time()

    if local_model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # breakpoint()
        # Apply inference mode: TensorRT or PyTorch
        if args.inference_mode == "tensorrt":
            logging.info(f"Replacing DiT with TensorRT engine: {args.trt_engine_path}")
            replace_dit_with_tensorrt(policy, args.trt_engine_path)
            logging.info(" TensorRT mode enabled")
        else:
            # PyTorch mode — skip torch.compile when sweeping action_horizon
            # (torch.compile + max-autotune caches kernel shapes, causing CUDA
            #  assert failures when action_horizon changes across iterations)
            logging.info(" PyTorch mode enabled (eager, no torch.compile)")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    else:
        assert 0, "Please provide valid model_path argument for inference"
    # Override model action_horizon if requested
    # 需要同时修改三个地方：
    # 1. model config 和 action_head（控制噪声初始化和扩散生成的步数）
    # 2. processor 的 modality_configs 中 action 的 delta_indices（控制 decode_action 的截断长度）
    # 3. processor 的 max_action_horizon（控制 action padding 上限，需 >= action_horizon）
    if args.model_action_horizon is not None:
        # breakpoint()
        new_ah = args.model_action_horizon
        # old_ah = policy.model.config.action_horizon
        # 模型层：修改 config 和 action_head
        policy.model.config.action_horizon = new_ah
        policy.model.action_head.action_horizon = new_ah
        # 扩展 position_embedding（nn.Embedding(1024, 1536)）以支持 action_horizon > 1024
        if hasattr(policy.model.action_head, 'position_embedding') and new_ah > policy.model.config.max_seq_len:
            import torch.nn as nn
            old_emb = policy.model.action_head.position_embedding
            new_emb = nn.Embedding(new_ah, old_emb.embedding_dim).to(
                device=old_emb.weight.device, dtype=old_emb.weight.dtype
            )
            nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
            new_emb.weight.data[:old_emb.num_embeddings] = old_emb.weight.data
            policy.model.action_head.position_embedding = new_emb
            policy.model.config.max_seq_len = new_ah
            logging.info(f"Extended position_embedding: 1024 -> {new_ah}")
        # Processor 层：修改 delta_indices 使 decode_action 不截断
        embodiment_tag = policy.embodiment_tag.value
        policy.processor.modality_configs[embodiment_tag]["action"].delta_indices = list(range(new_ah))
        # Processor 层：确保 max_action_horizon >= new_ah
        if policy.processor.max_action_horizon < new_ah:
            policy.processor.max_action_horizon = new_ah
        # 归一化参数：norm_params 存储为 (16, dim)，需压缩为 (dim,) 以支持任意 action_horizon 广播
        norm_params = policy.processor.state_action_processor.norm_params
        if embodiment_tag in norm_params and "action" in norm_params[embodiment_tag]:
            for joint_key, joint_params in norm_params[embodiment_tag]["action"].items():
                if joint_key == "dim":
                    continue
                for param_name in ["min", "max", "mean", "std"]:
                    if param_name in joint_params and hasattr(joint_params[param_name], 'ndim') and joint_params[param_name].ndim == 2:
                        joint_params[param_name] = joint_params[param_name][0]  # (16, D) -> (D,)
        # logging.info(f"Overriding model action_horizon: {old_ah} -> {new_ah}")

    model_load_time = time.time() - model_load_start
    logging.info(f"Model loading time: {model_load_time:.4f} seconds")

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Dataset creation
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 2: Creating Dataset Loader ===")
    logging.info("=" * 80)
    dataset_load_start = time.time()

    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    dataset_load_time = time.time() - dataset_load_start
    logging.info(f"Dataset loader creation time: {dataset_load_time:.4f} seconds")

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    # Evaluation loop
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 3: Running Evaluation ===")
    logging.info("=" * 80)

    all_mse = []
    all_mae = []
    all_timings = []
    pred_actions = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        (
            state_keys,
            action_keys,
            pred_action_across_time,
            traj,
            actual_steps,
            timing_dict,
            obs,
        ) = run_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            steps=args.steps,
            action_horizon=args.action_horizon,
            skip_timing_steps=args.skip_timing_steps,
        )
        print(f"查看最终结果：{type(pred_action_across_time)}")
        pred_actions.append(pred_action_across_time)

        if args.get_performance_stats:
            mse, mae = evaluate_predictions(
                state_keys,
                action_keys,
                pred_action_across_time,
                traj,
                traj_id,
                actual_steps,
                args.action_horizon,
                save_plot_path=None,
            )

            logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
            all_mse.append(mse)
            all_mae.append(mae)
            all_timings.append(timing_dict)

    if args.get_performance_stats:
        # Final performance summary
        logging.info("\n" + "=" * 80)
        logging.info("=== EVALUATION SUMMARY ===")
        logging.info("=" * 80)

        if all_mse:
            avg_mse = np.mean(np.array(all_mse))
            avg_mae = np.mean(np.array(all_mae))
            logging.info("\nMetrics:")
            logging.info(f"  Average MSE across all trajs: {avg_mse:.6f}")
            logging.info(f"  Average MAE across all trajs: {avg_mae:.6f}")
        else:
            logging.info("No valid trajectories were evaluated.")

        # Detailed timing summary
        logging.info("\n" + "=" * 80)
        logging.info("=== DETAILED TIMING SUMMARY ===")
        logging.info("=" * 80)
        logging.info("\nInitialization:")
        logging.info(f"  Model loading time:          {model_load_time*1000:.4f}ms")
        logging.info(f"  Dataset loader creation:     {dataset_load_time*1000:.4f}ms")

        total_inference = 0.0
        if all_timings:
            # Aggregate timing statistics
            total_episode_load = sum(t["episode_load_time"] for t in all_timings)
            total_data_prep = sum(sum(t["data_prep_times"]) for t in all_timings)
            total_inference = sum(sum(t["inference_times"]) for t in all_timings)

            # Count total inference steps
            total_inference_steps = sum(len(t["inference_times"]) for t in all_timings)

            logging.info(f"\nPer-Trajectory Timings ({len(all_timings)} trajectories):")
            logging.info(
                f"  Total episode loading:       {total_episode_load*1000:.4f}ms  (avg: {total_episode_load / len(all_timings)*1000:.4f}ms)"
            )

            if total_inference_steps > 0:
                logging.info(
                    f"  Total data preparation:      {total_data_prep*1000:.4f}ms  (avg: {total_data_prep / total_inference_steps*1000:.4f}ms per step)"
                )
                logging.info(
                    f"  Total inference:             {total_inference*1000:.4f}ms  (avg: {total_inference / total_inference_steps*1000:.4f}ms per step)"
                )

                logging.info("\nInference Statistics (last step only):")
                logging.info(f"  Total inference steps:       {total_inference_steps}")
                logging.info(
                    f"  Avg inference time per step: {total_inference / total_inference_steps*1000:.4f}ms"
                )

                # Collect all inference times for min/max/p90
                all_inf_times = [t for timing in all_timings for t in timing["inference_times"]]
                logging.info(f"  Min inference time:          {min(all_inf_times)*1000:.4f}ms")
                logging.info(f"  Max inference time:          {max(all_inf_times)*1000:.4f}ms")
                logging.info(f"  P90 inference time:          {np.percentile(all_inf_times, 90)*1000:.4f}ms")
            else:
                logging.warning("  No inference steps were timed (all skipped).")

    logging.info("=" * 80)
    logging.info("Done")
    return total_inference, pred_actions, obs


if __name__ == "__main__":
    import gc
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    CHUNK_SIZE = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    totals = []
    for chunk in CHUNK_SIZE:
        config.model_action_horizon = chunk
        config.action_horizon = chunk
        total, *_ = main(config)
        totals.append(total)
        # Free GPU memory between iterations
        gc.collect()
        torch.cuda.empty_cache()
        print(f"=== chunk={chunk} -> inference_time={total*1000:.2f}ms ===")

    print("\n" + "=" * 80)
    print("SWEEP RESULTS:")
    for chunk, t in zip(CHUNK_SIZE, totals):
        print(f"  action_horizon={chunk:>5d}  inference_time={t*1000:.2f}ms")
    print("=" * 80)
