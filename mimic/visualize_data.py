import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer
from huggingface_hub import hf_hub_download
import pathlib

HERE = pathlib.Path(__file__).parent

def reconstruct_qpos_from_imitation_data(observations, robot="booster_lower_t1"):
    """
    Reconstruct qpos from imitation learning observations.
    Observations contain robot_qpos in first 12 dimensions.
    For lower_t1: qpos = [root_pos(3), root_quat(4), joint_pos(12)] = 19D
    """
    robot_qpos = observations[:, :12]  # Extract joint positions
    
    # Default root position and orientation for lower_t1
    # Root position: typically at origin with some height
    root_pos = np.array([0.0, 0.0, 0.675])  # Default height for lower_t1
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (w, x, y, z)
    
    # Construct full qpos: [root_pos(3), root_quat(4), joint_pos(12)]
    T = robot_qpos.shape[0]
    qpos_traj = np.zeros((T, 19), dtype=np.float32)
    qpos_traj[:, 0:3] = root_pos  # Root position
    qpos_traj[:, 3:7] = root_quat  # Root orientation
    qpos_traj[:, 7:19] = robot_qpos  # Joint positions
    
    return qpos_traj

def main():
    parser = argparse.ArgumentParser(description="Play qpos from NPZ in MuJoCo.")
    parser.add_argument(
        "--robot",
        choices=["booster_t1", "booster_lower_t1"],
        default="booster_t1",
    )
    parser.add_argument("--npz", required=True, help="Path to .npz with qpos (T, nq) or imitation learning data.")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (overrides any 'fps' in the NPZ).")
    parser.add_argument("--episode", type=int, default=None, help="Episode index to visualize (for imitation learning data).")
    args = parser.parse_args()

    # --- Load trajectory ---
    try:
        data_npz = np.load(args.npz, allow_pickle=True)
    except:
        file_name = hf_hub_download(
                    repo_id="SaiResearch/booster_dataset",
                    filename=f"soccer/{args.robot}/{args.npz}",
                    repo_type="dataset")
        data_npz = np.load(file_name, allow_pickle=True)
    
    # Check if this is imitation learning data format
    is_imitation_data = "observations" in data_npz and "actions" in data_npz
    
    if is_imitation_data:
        print("Detected imitation learning data format")
        observations = np.array(data_npz["observations"], dtype=np.float32)
        done = np.array(data_npz["done"], dtype=bool) if "done" in data_npz else None
        
        # Extract episode if specified
        if args.episode is not None and done is not None:
            # Find episode boundaries
            episode_starts = np.where(np.concatenate([[True], done[:-1]]))[0]
            episode_ends = np.where(done)[0] + 1
            
            if args.episode >= len(episode_starts):
                raise ValueError(f"Episode {args.episode} not found. Available episodes: 0-{len(episode_starts)-1}")
            
            start_idx = episode_starts[args.episode]
            end_idx = episode_ends[args.episode] if args.episode < len(episode_ends) else len(observations)
            observations = observations[start_idx:end_idx]
            print(f"Visualizing episode {args.episode}: steps {start_idx} to {end_idx}")
        
        # Reconstruct qpos from observations
        qpos_traj = reconstruct_qpos_from_imitation_data(observations, args.robot)
        print(f"Reconstructed qpos trajectory: {qpos_traj.shape}")
    else:
        # Original format with qpos
        key = "qpos"
        if key not in data_npz:
            raise KeyError(f"'{key}' not found in {args.npz}. Available: {list(data_npz.keys())}")
        
        qpos_traj = np.array(data_npz[key], dtype=float)  # (T, nq)

    if qpos_traj.ndim != 2:
        raise ValueError(f"qpos must be 2D (T, nq). Got shape {qpos_traj.shape}")

    # Optional fps in file
    file_fps = float(data_npz["fps"]) if ("fps" in data_npz and args.fps is None) else None
    fps = args.fps if args.fps is not None else (file_fps if file_fps and file_fps > 0 else 30.0)
    dt_frame = 1.0 / fps

    # --- Load model & data ---
    model = mujoco.MjModel.from_xml_path(f"{HERE}/assets/booster_t1/{args.robot}.xml")
    data = mujoco.MjData(model)

    T, nq = qpos_traj.shape
    if nq != model.nq:
        raise ValueError(f"qpos width ({nq}) != model.nq ({model.nq}).")

    # Start from first pose
    data.qpos[:] = qpos_traj[0]
    mujoco.mj_forward(model, data)

    # --- Launch viewer and play ---
    print(f"Playing {T} frames at {fps:.2f} FPS...")
    start_time = time.time()

    with viewer.launch_passive(model, data) as v:
        print(T)
        for t in range(T):

            # Set pose and forward
            data.qpos[:] = qpos_traj[t]
            mujoco.mj_forward(model, data)

            # Render & pace to FPS
            v.sync()
            # Wall-clock pacing (simple)
            target = start_time + (t + 1) * dt_frame
            now = time.time()
            if target > now:
                time.sleep(target - now)

    print("Done.")

if __name__ == "__main__":
    main()
 