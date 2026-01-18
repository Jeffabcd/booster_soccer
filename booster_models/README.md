---
license: mit
language:
- en
pipeline_tag: reinforcement-learning
tags:
- robotics
- reinforcement_learning
- humanoid
- soccer
- sai
- mujoco
---

## Model Details

### Model Description

This repository hosts the **Booster Soccer Controller Suite** — a collection of reinforcement learning policies and controllers powering humanoid agents in the [**Booster Soccer Showdown**](https://competesai.com/competitions/cmp_xnSCxcJXQclQ).

It contains:
1. **Low-Level Controller (robot/):**  
   A proprioceptive policy for the **Lower T1** humanoid that converts high-level commands (forward, lateral, and yaw velocities) into joint angle targets.
2. **Competition Policies (model/):**  
   High-level agents trained in SAI’s soccer environments that output those high-level commands for match-time play.

- **Developed by:** ArenaX Labs  
- **License:** MIT  
- **Frameworks:** PyTorch · MuJoCo · Stable-Baselines3  
- **Environments:** Booster Gym / SAI Soccer tasks  


## Testing Instructions

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

### Teleoperation

Booster Soccer Showdown supports keyboard teleop out of the box.

```bash
python booster_control/teleoperate.py \
  --env LowerT1GoaliePenaltyKick-v0 
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

---

### Training

We provide a minimal reinforcement learning pipeline for training agents with **Deep Deterministic Policy Gradient (DDPG)** in the Booster Soccer Showdown environments in the `training_scripts/` folder. The training stack consists of three scripts:

#### 1) `ddpg.py`

Defines the **DDPG_FF model**, including:

* Actor and Critic neural networks with configurable hidden layers and activation functions.
* Target networks and soft-update mechanism for stability.
* Training step implementation (critic loss with MSE, actor loss with policy gradient).
* Utility functions for forward passes, action selection, and backpropagation.

---

#### 2) `training.py`

Provides the **training loop** and supporting components:

* **ReplayBuffer** for experience storage and sampling.
* **Exploration noise** injection to encourage policy exploration.
* Iterative training loop that:

  * Interacts with the environment.
  * Stores experiences.
  * Periodically samples minibatches to update actor/critic networks.
* Tracks and logs progress (episode rewards, critic/actor loss) with `tqdm`.

---

#### 3) `main.py`

Serves as the **entry point** to run training:

* Initializes the Booster Soccer Showdown environment via the **SAI client**.
* Defines a **Preprocessor** to normalize and concatenate robot state, ball state, and environment info into a training-ready observation vector.
* Instantiates a **DDPG_FF model** with custom architecture.
* Defines an **action function** that rescales raw policy outputs to environment-specific action bounds.
* Calls the training loop, and after training, supports:

  * `sai.watch(...)` for visualizing learned behavior.
  * `sai.benchmark(...)` for local benchmarking.

---

#### Example: Run Training

```bash
python training_scripts/main.py
```

This will:

1. Build the environment.
2. Initialize the model.
3. Run the training loop with replay buffer and DDPG updates.
4. Launch visualization and benchmarking after training.


#### Example: Test pretrained model

```bash
python training_scripts/test.py --env LowerT1KickToTarget-v0
```



