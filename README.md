# BOOSTER SOCCER SHOWDOWN

![Booster Soccer Showdown Banner](resources/comp.png)  

A fast, extensible robotics soccer competition focused on **generalization across environments**. All the models and datasets are hosted on [Hugging Face](https://huggingface.co/SaiResearch) and the competition is live on [SAI](https://competesai.com/competitions/cmp_xnSCxcJXQclQ).

---

## Compatibility & Requirements

* **Python**: 3.10+
* **Pip**: compatible with environments using Python ≥ 3.10
* **OS**: macOS (Apple Silicon), Linux, and Windows

> Tip: Use a Python 3.10+ environment created via `pyenv`, `conda`, or `uv` for the smoothest experience.

---

## Installation

1. **Clone the repo**

```bash
git clone https://github.com/ArenaX-Labs/booster_soccer_showdown.git
cd booster_soccer_showdown
```

2. **Create & activate a Python 3.10+ environment**

```bash
# any env manager is fine; here are a few options
# --- venv ---
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# --- conda ---
# conda create -n booster-ssl python=3.11 -y && conda activate booster-ssl
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Teleoperation

Booster Soccer Showdown supports keyboard teleop out of the box.

```bash
python booster_control/teleoperate.py \
  --env LowerT1GoaliePenaltyKick-v0 
  --renderer mujoco
```

**Default bindings (example):**

* `W/S`: move forward/backward
* `A/D`: move left/right
* `Q/E`: rotate left/right
* `L`: reset commands
* `P`: reset environment

---

⚠️ **Note for macOS and Windows users**
Because different renderers are used on macOS and Windows, you may need to adjust the **position** and **rotation** sensitivity for smooth teleoperation.
Run the following command with the sensitivity flags set explicitly:

```bash
python booster_control/teleoperate.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --pos_sensitivity 1.5 \
  --rot_sensitivity 1.5
```

(Tune `--pos_sensitivity` and `--rot_sensitivity` as needed for your setup.)

There is another renderer as well which can be used, which speeds up the simulation - 

```bash
python booster_control/teleoperate.py \
  --env LowerT1GoaliePenaltyKick-v0 \
  --renderer mjviewer
```
---

## Mimic

The `mimic/` tools let you replay and analyze motions with MuJoCo.

### What’s inside

* `mimic/forward_kinematics.py` — computes derived robot signals (end-effector poses, COM, contacts, velocities, etc.) from motion data using MuJoCo kinematics.
* `mimic/visualize_data.py` — replays a motion in the MuJoCo viewer at a chosen FPS.
* **Model**: the Booster T1 MuJoCo XML is in `mimic/assets/booster_t1/`.

> Motion files are hosted on [Hugging Face](https://huggingface.co/datasets/SaiResearch/booster_dataset) and can be downloaded to use with these scripts. 

---

### 1) Compute kinematics from a motion file

```bash
python mimic/forward_kinematics.py \
  --robot booster_t1 \
  --npz goal_kick.npz \
  --out out/example_motion_fk.npz
```

**Args (common):**

* `--robot` : robot to be loaded (choices: booster_t1 or booster_lower_t1)
* `--npz` : Name of the motion file (`.npz`).
* `--out` : output `.npz` file with enriched signals.

---

### 2) Visualize a motion in MuJoCo

```bash
# Visualize raw motion
python mimic/visualize_data.py \
  --robot booster_t1 \
  --npz goal_kick.npz \
  --fps 30
```
---

## Training

We provide a minimal reinforcement learning pipeline for training agents with **Deep Deterministic Policy Gradient (DDPG)** in the Booster Soccer Showdown environments in the `training_scripts/` folder. The training stack consists of three scripts:

### 1) `ddpg.py`

Defines the **DDPG_FF model**, including:

* Actor and Critic neural networks with configurable hidden layers and activation functions.
* Target networks and soft-update mechanism for stability.
* Training step implementation (critic loss with MSE, actor loss with policy gradient).
* Utility functions for forward passes, action selection, and backpropagation.

---

### 2) `training.py`

Provides the **training loop** and supporting components:

* **ReplayBuffer** for experience storage and sampling.
* **Exploration noise** injection to encourage policy exploration.
* Iterative training loop that:

  * Interacts with the environment.
  * Stores experiences.
  * Periodically samples minibatches to update actor/critic networks.
* Tracks and logs progress (episode rewards, critic/actor loss) with `tqdm`.

---

### 3) `main.py`

Serves as the **entry point** to run training:

* Initializes the Booster Soccer Showdown environment via the **SAI client**.
* Defines a **Preprocessor** to normalize and concatenate robot state, ball state, and environment info into a training-ready observation vector.
* Instantiates a **DDPG_FF model** with custom architecture.
* Defines an **action function** that rescales raw policy outputs to environment-specific action bounds.
* Calls the training loop, and after training, supports:

  * `sai.watch(...)` for visualizing learned behavior.
  * `sai.benchmark(...)` for local benchmarking.

---

### Example: Run Training

```bash
python training_scripts/main.py
```

This will:

1. Build the environment.
2. Initialize the model.
3. Run the training loop with replay buffer and DDPG updates.
4. Launch visualization and benchmarking after training.


### Example: Test pretrained model

```bash
python training_scripts/test.py --env LowerT1KickToTarget-v0
```

---

## Imitation Learning

An **imitation learning pipeline** designed for training robust agents to mimic expert demonstrations in the **Booster Soccer Showdown** environments. This repository supports data collection, preprocessing, model training, conversion between frameworks (JAX ↔ PyTorch), and submission-ready model packaging.

To make it easy to train models, the IL models are trained to output joint positions which are then converted to torque using a PD controller before feeding it to the simulator.

### Data Collection

You can collect teleoperation or scripted demonstration data using:

```bash
python imitation_learning/scripts/collect_data.py \
  --env LowerT1KickToTarget-v0 \
  --data_set_directory path/to/data.npz \
  --renderer mjviewer
```

This script records trajectories in `.npz` format containing observations and actions, rewards.

Data collection automatically includes preprocessing to ensure consistent observation spaces across all environments. This is done through the built-in `Preprocessor` class in the `imitation_learning/scripts/preprocessor.py` script, which augments each observation with projected gravity and base angular velocity derived from robot state information. It can be modified according to the requirement of the user.

### Training Imitation Learning Agents

Train an imitation learning agent (e.g., BC, IQL, HIQL) end-to-end:

```bash
python imitation_learning/train.py \
  --agents bc \
  --dataset_dir path/to/data.npz
```

Supported agents:

* `bc` — Behavioral Cloning
* `iql` — Implicit Q-Learning
* `gbc` — Goal-Conditioned BC - experimental
* `hiql` — Hierarchical Imitation Q-Learning - experimental
* `gqicl` — Goal-Conditioned IQL - experimental

The checkpoints are saved in the `./exp` folder by default.

### Evaluation

Test your trained policy in the simulator:

```bash
python imitation_learning/test.py \
  --restore_path path/to/checkpoints \
  --restore_epoch 1000000 \
  --dataset_dir path/to/data.npz
```

### Model Conversion (JAX → PyTorch)

If your model was trained in JAX/Flax, convert it to PyTorch for submission on SAI:

```bash
python imitation_learning/scripts/jax2torch.py \
  --pkl path/to/checkpoint.pkl \
  --out path/to/model.pt
```

### Submission

To submit the converted model on SAI:

```bash
python imitation_learning/submission/submit_sai.py 
```

---

## Diffusion Policy

The repository includes a **Diffusion Policy** implementation for training robust imitation learning agents using diffusion models. Diffusion policies predict sequences of actions conditioned on observation history, providing smooth and stable control.

### Configuration Setup

Before training, you need to configure the dataset path and task parameters:

#### 1) Configure Task Settings

Edit `diffusion_policy/diffusion_policy/config/task/booster_lowdim.yaml`:

```yaml
name: booster_lowdim

obs_dim: 89          # Observation dimension (matches preprocessed observations)
action_dim: 12       # Action dimension (normalized joint actions)
keypoint_dim: 2      # Not used but required by some configs

dataset:
  _target_: diffusion_policy.dataset.booster_lowdim_dataset.BoosterLowdimDataset
  dataset_path: ../booster_dataset/imitation_learning/booster_soccer_showdown.npz  # Update this path
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.05
```

**Important**: Update the `dataset_path` to point to your collected dataset (`.npz` file).

#### 2) Configure Training Settings

Edit `diffusion_policy/diffusion_policy/config/train_diffusion_unet_booster_lowdim.yaml` to adjust training parameters:

Key parameters you may want to modify:

```yaml
horizon: 16              # Prediction horizon
n_obs_steps: 2           # Number of observation steps to condition on
n_action_steps: 8        # Number of action steps to predict
n_latency_steps: 0       # Latency compensation

training:
  device: "cuda:0"       # Training device
  num_epochs: 3000       # Number of training epochs
  checkpoint_every: 100  # Save checkpoint every N epochs
  val_every: 10          # Run validation every N epochs

dataloader:
  batch_size: 256        # Training batch size
  num_workers: 4         # Data loading workers

optimizer:
  lr: 1.0e-4            # Learning rate
```

### Training

Train a diffusion policy using the provided script:

```bash
./train_diffusion.sh
```

Or manually:

```bash
cd diffusion_policy

python train.py \
  --config-name=train_diffusion_unet_booster_lowdim \
  training.device=cuda:0 \
  training.num_epochs=3000 \
  dataloader.batch_size=256 \
  dataloader.num_workers=4
```

**Alternative: Diffusion Transformer**

You can also train using a Transformer-based diffusion policy:

```bash
cd diffusion_policy

python train.py \
  --config-name=train_diffusion_transformer_booster_lowdim \
  training.device=cuda:0 \
  training.num_epochs=3000 \
  dataloader.batch_size=256
```

Checkpoints are saved in `diffusion_policy/data/outputs/` with timestamps.

### Testing

Test your trained diffusion policy checkpoint:

```bash
python imitation_learning/test_diffusion.py \
  --checkpoint diffusion_policy/data/outputs/YYYY.MM.DD/HH.MM.SS_train_diffusion_unet_booster_lowdim_booster_lowdim/checkpoints/latest.ckpt \
  --env_name LowerT1GoaliePenaltyKick-v0 \
  --action_scale 1.0 \
  --episodes 5
```

**Arguments:**

* `--checkpoint`: Path to the trained checkpoint (`.ckpt` file)
* `--env_name`: Environment to test in (e.g., `LowerT1GoaliePenaltyKick-v0`)
* `--action_scale`: Scale factor for actions (default: 1.0, try 1.5-2.0 if actions are too conservative)
* `--episodes`: Number of episodes to run (default: 5)
* `--render`: Enable rendering (default: True)
* `--renderer`: Renderer to use (`mujoco` or `mjviewer`, default: `mujoco`)
* `--test_random`: Test with random actions instead of policy (for debugging)

**Example with shell script:**

```bash
./test_diffusion.sh
```

The test script will output:
* Episode returns (mean and std)
* Step counts per episode (mean and std)
* Success rates (mean and std)

### How It Works

1. **Observation Processing**: The policy uses a sliding window of `n_obs_steps` observations (default: 2) to condition action prediction.

2. **Action Prediction**: The diffusion model predicts a sequence of `n_action_steps` actions (default: 8) at once.

3. **Action Execution**: Actions are executed sequentially from the predicted sequence. When all actions are used, a new prediction is made.

4. **Action Conversion**: The policy outputs normalized actions (range [-1, 1]) which are converted to joint torques using `LowerT1JoyStick.get_torque()` before being applied to the environment.

### Demo
![demo](diffusion_policy_booster.gif)
### Tips

* If episodes terminate quickly, check that the observation buffer is properly initialized after reset
* The policy uses observation normalization - ensure your dataset matches the expected observation format
* For faster training, reduce `num_epochs` or increase `batch_size` (if GPU memory allows)
