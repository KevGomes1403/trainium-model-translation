"""
Closed-loop LIBERO demo: SmolVLA on Trainium drives a sim arm to do a task.

What you'll see in the output MP4:
  - The LIBERO mujoco scene (agentview camera): the Franka arm moves through
    the world, grasping or attempting to grasp a target object as instructed.
  - Side panel: the wrist-camera view + current task description + step
    counter + per-frame inference latency.
  - The arm is driven step-by-step by the Neuron-compiled SmolVLA pipeline:
        (agentview, wrist, empty) -> Vision NEFF (3 NEFF calls)
        + state + tokenized instruction -> Prefix NEFF
        -> 10 Euler steps of Denoise NEFF -> 50-step action chunk
        -> execute first ACTION_CHUNK_K actions, replan, repeat.

Inputs come straight from mujoco render at every step. This is closed-loop:
the model is *driving* the simulation. The arm visibly moves toward and
grasps objects when the policy is good.

Usage:
    python -m smol_vla.demo_libero \\
        --hf-checkpoint /path/to/lerobot/smolvla_libero \\
        --neff-dir      /path/to/smol_vla_neff_libero \\
        --suite libero_object --task-id 0 \\
        --output /path/to/output.mp4 --max-steps 280

Tasks of interest (libero_object):
  0  pick up the alphabet soup and place it in the basket
  1  pick up the cream cheese and place it in the basket
  2  pick up the salad dressing and place it in the basket
  ...  (10 tasks total in libero_object)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering

try:
    from . import config_constants as C
    from .modeling_smolvla import SmolVLAPolicy
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from smol_vla import config_constants as C
    from smol_vla.modeling_smolvla import SmolVLAPolicy


STATE_DIM_LIBERO = 8        # 3 eef_pos + 3 eef_axis_angle + 2 gripper_qpos
ACTION_DIM_LIBERO = 7       # 3 dxyz + 3 daxis-angle + 1 gripper


# ---------------------------------------------------------------------------
# Image preprocessing — same resize_with_pad as the SO100 demo
# ---------------------------------------------------------------------------

def _resize_with_pad(img_hwc_uint8: np.ndarray, size: int = 512,
                     pad_value: float = 0.0) -> np.ndarray:
    """Match lerobot.policies.smolvla.modeling_smolvla.resize_with_pad exactly:
    keep aspect ratio, then pad on the LEFT and TOP only with `pad_value=0`,
    then map [0,1] -> [-1,1]. Image lands in the BOTTOM-RIGHT corner.
    """
    img = img_hwc_uint8.astype(np.float32) / 255.0
    h0, w0 = img.shape[:2]
    ratio = max(w0 / size, h0 / size)
    new_w, new_h = int(w0 / ratio), int(h0 / ratio)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    t = t.squeeze(0)
    out = torch.full((3, size, size), pad_value, dtype=torch.float32)
    pad_h = size - new_h
    pad_w = size - new_w
    out[:, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = t
    return (out * 2.0 - 1.0).numpy()  # [-1, 1]


# ---------------------------------------------------------------------------
# State construction (LIBERO obs -> 8-D state vector matching the finetune)
# ---------------------------------------------------------------------------

def _quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternion to axis-angle vector (axis * angle)."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat shape {q.shape}")
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    x, y, z, w = q
    angle = 2.0 * np.arctan2(np.sqrt(x*x + y*y + z*z), w)
    s = np.sqrt(x*x + y*y + z*z)
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z]) / s
    return (axis * angle).astype(np.float32)


def _build_libero_state(obs: dict) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)            # [3]
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)          # [4] xyzw
    aa = _quat_to_axis_angle(eef_quat)                                        # [3]
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)  # [2]
    return np.concatenate([eef_pos, aa, gripper_qpos], axis=0)                # [8]


# ---------------------------------------------------------------------------
# Normalization (using the libero finetune's stats)
# ---------------------------------------------------------------------------

class LiberoNormalizer:
    """Normalize state, denormalize action, using stats from the finetune ckpt."""

    def __init__(self, checkpoint_dir: str):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(
            checkpoint_dir,
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
        ))
        self.state_mean = sd["observation.state.mean"].numpy().astype(np.float32)  # [8]
        self.state_std  = sd["observation.state.std"].numpy().astype(np.float32)
        self.action_mean = sd["action.mean"].numpy().astype(np.float32)             # [7]
        self.action_std  = sd["action.std"].numpy().astype(np.float32)

    def state(self, raw_state: np.ndarray) -> np.ndarray:
        # raw_state shape (8,) -> normalized 32-d (zero-padded)
        norm = (raw_state - self.state_mean) / (self.state_std + 1e-8)
        out = np.zeros(C.MAX_STATE_DIM, dtype=np.float32)
        out[:STATE_DIM_LIBERO] = norm
        return out

    def action(self, normalized_chunk_32: np.ndarray) -> np.ndarray:
        # input [50, 32] normalized -> [50, 7] denormalized
        first7 = normalized_chunk_32[:, :ACTION_DIM_LIBERO]
        return first7 * self.action_std + self.action_mean


# ---------------------------------------------------------------------------
# Closed-loop runner
# ---------------------------------------------------------------------------

class ClosedLoopRunner:
    def __init__(self, hf_checkpoint: str, neff_dir: str):
        print("Loading 3 NEFFs to Neuron ...")
        self.policy = SmolVLAPolicy(hf_checkpoint_dir=hf_checkpoint, tp_degree=1)
        self.policy.load(neff_dir)
        self.normalizer = LiberoNormalizer(hf_checkpoint)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Instruct"
        )

    def _build_inputs(self, agentview_uint8: np.ndarray, wrist_uint8: np.ndarray,
                       state_8: np.ndarray, lang_token_ids: torch.Tensor,
                       lang_mask: torch.Tensor = None):
        cam1 = _resize_with_pad(agentview_uint8)   # [3, 512, 512]
        cam2 = _resize_with_pad(wrist_uint8)
        # HuggingFaceVLA fork: 2 cameras (image, image2). No dummy 3rd cam.
        images = [
            torch.from_numpy(cam1).unsqueeze(0).to(torch.bfloat16),
            torch.from_numpy(cam2).unsqueeze(0).to(torch.bfloat16),
        ]
        norm_state = self.normalizer.state(state_8)
        state_t = torch.from_numpy(norm_state).unsqueeze(0).to(torch.float32)
        return images, lang_token_ids, state_t

    def predict_chunk(self, agentview_uint8: np.ndarray, wrist_uint8: np.ndarray,
                       state_8: np.ndarray, lang_token_ids: torch.Tensor,
                       lang_mask: torch.Tensor = None) -> np.ndarray:
        """Return [50, 7] denormalized actions ready for env.step()."""
        images, lang, state_t = self._build_inputs(
            agentview_uint8, wrist_uint8, state_8, lang_token_ids,
        )
        chunk_32 = self.policy.generate(images, lang, state_t, lang_mask=lang_mask)
        chunk_32 = chunk_32.squeeze(0).cpu().numpy()                       # [50, 32]
        actions = self.normalizer.action(chunk_32)                         # [50, 7]
        # LIBERO env clips at [-1, 1]
        return np.clip(actions, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Animation rendering — agentview frames + side panel
# ---------------------------------------------------------------------------

def _render_video(output_path: str,
                  agentview_frames: List[np.ndarray],
                  wrist_frames: List[np.ndarray],
                  task_text: str,
                  per_step_meta: List[dict],
                  fps: int = 20,
                  suite_name: str = "",
                  task_id: int = 0,
                  seed: int = 0):
    """Clean, professional rendering: large agentview, small wrist inset,
    minimal HUD with task / step / latency / status."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import imageio_ffmpeg

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["font.family"] = ["DejaVu Sans"]

    T = len(agentview_frames)
    BG = "#0e1116"     # deep slate background
    FG = "#e6edf3"     # near-white text
    MUTED = "#8b949e"  # muted gray
    ACCENT = "#58a6ff" # cool blue accent
    GOOD = "#3fb950"   # success green

    # 16:9 figure, agentview takes most of the space.
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Main agentview, full canvas
    ax_main = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_main.set_facecolor(BG)
    ax_main.set_xticks([]); ax_main.set_yticks([])
    for s in ax_main.spines.values():
        s.set_visible(False)
    im_main = ax_main.imshow(agentview_frames[0])

    # Wrist inset (top-right), framed
    ax_wrist = fig.add_axes([0.785, 0.625, 0.20, 0.355])
    ax_wrist.set_facecolor(BG)
    ax_wrist.set_xticks([]); ax_wrist.set_yticks([])
    for s in ax_wrist.spines.values():
        s.set_color(ACCENT); s.set_linewidth(1.2)
    im_wrist = ax_wrist.imshow(wrist_frames[0])
    ax_wrist.set_title("wrist", color=MUTED, fontsize=9, pad=4, loc="left")

    # Top-left header: title + task
    fig.text(0.018, 0.965,
             "SmolVLA on AWS Trainium",
             color=FG, fontsize=14, fontweight="bold", family="DejaVu Sans")
    fig.text(0.018, 0.935,
             f"{suite_name}  ·  task {task_id}  ·  seed {seed}",
             color=MUTED, fontsize=9.5)
    fig.text(0.018, 0.905, f'"{task_text.strip()}"',
             color=ACCENT, fontsize=11.5, fontstyle="italic")

    # Bottom HUD: step / latency / status
    step_label   = fig.text(0.018, 0.055, "", color=FG,    fontsize=11, family="monospace")
    lat_label    = fig.text(0.018, 0.030, "", color=MUTED, fontsize=10, family="monospace")
    status_label = fig.text(0.982, 0.055, "", color=FG,    fontsize=11, family="monospace",
                            ha="right", fontweight="bold")

    # Subtle grip indicator: small horizontal bar bottom-center showing -1..+1
    bar_left, bar_y, bar_w, bar_h = 0.385, 0.030, 0.23, 0.012
    fig.add_artist(plt.Rectangle((bar_left, bar_y), bar_w, bar_h,
                                  facecolor="#21262d", edgecolor=MUTED,
                                  transform=fig.transFigure, lw=0.5))
    grip_marker = fig.add_artist(plt.Rectangle((bar_left + bar_w/2, bar_y), 0.005, bar_h,
                                                facecolor=ACCENT, transform=fig.transFigure))
    fig.text(bar_left,        0.012, "close", color=MUTED, fontsize=8)
    fig.text(bar_left + bar_w, 0.012, "open",  color=MUTED, fontsize=8, ha="right")
    fig.text(bar_left + bar_w/2, 0.060, "gripper",
             color=MUTED, fontsize=8.5, ha="center")

    def update(i):
        im_main.set_data(agentview_frames[i])
        im_wrist.set_data(wrist_frames[i])
        m = per_step_meta[i]

        step_label.set_text(f"step  {i + 1:>3d} / {T}")
        lat_label.set_text(f"infer {m.get('latency_ms', 0):>5.1f} ms  (Neuron)")

        success = m.get("success", False)
        if success:
            status_label.set_text("✓  SUCCESS")
            status_label.set_color(GOOD)
        else:
            status_label.set_text("• running")
            status_label.set_color(MUTED)

        # gripper marker: action[6] in [-1, 1] → x in bar
        g = float(np.clip(m["action"][6], -1.0, 1.0))
        gx = bar_left + bar_w * (g + 1.0) / 2.0
        grip_marker.set_x(gx - 0.0025)

        return [im_main, im_wrist, step_label, lat_label, status_label, grip_marker]

    anim = FuncAnimation(fig, update, frames=T, blit=False, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
    anim.save(output_path, writer=writer, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Closed-loop LIBERO demo with Neuron SmolVLA")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--neff-dir", required=True)
    parser.add_argument("--suite", default="libero_object",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--output", default="libero_demo.mp4")
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--replan-every", type=int, default=10,
                        help="Number of chunk actions to execute before replanning")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 1. Build LIBERO env via lerobot's wrapper — handles delta control mode +
    # observation format identically to how the dataset was recorded.
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_task_init_states
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    print(f"Suite: {args.suite}   Task {args.task_id}: {task.language}")
    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        camera_name=["agentview_image", "robot0_eye_in_hand_image"],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=args.max_steps,
        episode_index=args.seed,   # picks initial state
        control_mode="relative",
    )
    obs, _ = env.reset(seed=args.seed)

    # 2. Load policy
    runner = ClosedLoopRunner(args.hf_checkpoint, args.neff_dir)
    # Match lerobot's smolvla_new_line_processor: append a newline to the
    # task before tokenizing.
    enc = runner.tokenizer(
        task.language + "\n",
        max_length=C.NUM_TEXT_TOKENS, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    lang_token_ids = enc["input_ids"].to(torch.int32)
    lang_mask = enc["attention_mask"].bool()

    # 3. Closed-loop rollout
    print(f"Rolling out for up to {args.max_steps} steps (replan every {args.replan_every}) ...")
    agentview_frames = []
    wrist_frames = []
    per_step = []
    success = False

    chunk = None
    chunk_idx = 0

    for step in range(args.max_steps):
        # lerobot LiberoEnv obs format: {"pixels": {"image":..., "image2":...},
        #                               "robot_state": {...}}
        # 180° flip per lerobot.processor.env_processor:59 — accounts for
        # HuggingFaceVLA/libero camera orientation convention.
        agentview_flipped = obs["pixels"]["image"][::-1, ::-1].copy()
        wrist_flipped = obs["pixels"]["image2"][::-1, ::-1].copy()
        agentview_raw = agentview_flipped
        wrist_raw = wrist_flipped
        agentview_vis = agentview_flipped
        wrist_vis = wrist_flipped
        # Construct 8-D state from lerobot's robot_state dict
        rs = obs["robot_state"]
        state8 = np.concatenate([
            np.asarray(rs["eef"]["pos"], dtype=np.float32),                       # 3
            _quat_to_axis_angle(np.asarray(rs["eef"]["quat"], dtype=np.float32)), # 3
            np.asarray(rs["gripper"]["qpos"], dtype=np.float32),                  # 2
        ])
        agentview_frames.append(agentview_vis.copy())
        wrist_frames.append(wrist_vis.copy())

        # Replan?
        latency_ms = 0.0
        if chunk is None or chunk_idx >= args.replan_every:
            t0 = time.time()
            chunk = runner.predict_chunk(agentview_raw, wrist_raw, state8, lang_token_ids, lang_mask=lang_mask)
            latency_ms = (time.time() - t0) * 1000
            chunk_idx = 0

        action = chunk[chunk_idx].astype(np.float32)
        chunk_idx += 1
        obs, reward, terminated, truncated, info = env.step(action)
        success = bool(info.get("is_success", False))
        per_step.append({"action": action, "latency_ms": latency_ms, "success": success})
        if step % 20 == 0:
            print(f"  step {step:3d}  latency={latency_ms:6.1f}ms  action[0..2]=({action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f})  grip={action[6]:+.2f}  success={success}", flush=True)
        if success:
            print(f"  >>> SUCCESS at step {step}")
            break

    print(f"\nRollout finished. steps={len(per_step)}  success={success}")
    env.close()

    # 4. Render video
    _render_video(args.output, agentview_frames, wrist_frames,
                  task_text=task.language, per_step_meta=per_step, fps=20,
                  suite_name=args.suite, task_id=args.task_id, seed=args.seed)


if __name__ == "__main__":
    main()
