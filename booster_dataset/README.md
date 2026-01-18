---
license: mit
task_categories:
- robotics
language:
- en
tags:
- robotics
- humanoid
- reinforcement_learning
size_categories:
- n<1K
---
# Booster Dataset

The **Booster Dataset** is a collection of motion and control data capturing different humanoid robots performing a diverse set of soccer-related actions. These include skills necessary for robot soccer such as kicking, dribbling, and goal kicks.  

This dataset is designed to support research in **robot soccer, reinforcement learning, motion planning, imitation learning, and control of bipedal robots** in dynamic, contact-rich environments.

---

## Dataset Details

### Dataset Description

- **Curated by:** ArenaX Labs  
- **License:** MIT  
- **Format:** `.npz` files containing robot kinematic and dynamic states  
- **Purpose:** Provide expert demonstrations and trajectories for training and benchmarking soccer-playing humanoid robots.

---

## Uses

### Direct Use

- Training reinforcement learning and imitation learning policies.  
- Motion planning and control benchmarking for humanoid soccer.  
- Studying dynamic skills like ball-kicking, goal-kicking, and repositioning.  
- Curriculum learning: starting from balance and stepping, progressing to soccer maneuvers.  

### Out-of-Scope Use

- Human motion modeling or biomechanical studies (data is robot-specific).  
- Applications outside robotics locomotion and soccer (e.g., medical or sensitive domains).  
- Any use that attempts to infer personal, demographic, or identity-related data (not present in this dataset).  
---

## Dataset Structure

Each `.npz` file contains the following arrays:

- **qpos**: Concatenated positions (root position, root orientation quaternion, and DOF positions).  
- **qvel**: Concatenated velocities (linear velocity, angular velocity, and DOF velocities).  
- **xpos, xquat, cvel, subtree_com, site_xpos, site_xmat**: Currently placeholder arrays (`zeros`) reserved for extended features such as body/site positions and COM.  
- **split_points**: Start and end indices for trajectory segmentation.  
- **joint_names**: Names of robot joints.  
- **frequency**: Target control frequency of the recorded trajectory.  
- **njnt**: Number of joints.  
- **jnt_type**: Joint types (0 = root, 3 = hinge).  
- **body_names, site_names, metadata**: Reserved metadata placeholders.  
- **body_*** and **site_*** arrays: Empty placeholders for MuJoCo-style body/site information (position, orientation, weld IDs, etc.).  

This structure allows loading trajectories directly into MuJoCo-compatible formats for playback or analysis.

---

## Dataset Creation

### Curation Rationale

The dataset was created to provide a standardized benchmark of soccer-related skills for humanoid robots, facilitating progress in robotic soccer research.
 
### Recommendations
- Use as a **benchmark for policy learning** rather than as a standalone dataset for generalization.  
- Combine with simulated data augmentation for robustness.  

---

## Citation

If you use this dataset, please cite as:

**BibTeX**
```bibtex
@dataset{arenax2025booster,
  title     = {Booster Dataset},
  author    = {ArenaX Labs},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/SaiResearch/booster_dataset}
}
