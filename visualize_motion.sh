#!/bin/bash

# Visualize raw motion (original format with qpos)
# python mimic/visualize_data.py \
#   --robot booster_t1 \
#   --npz booster_dataset/soccer/booster_t1/jogging.npz \
#   --fps 30

# Visualize imitation learning data (preprocessed observations)
# Note: The robot will be visualized at a fixed position, but joint movements will be correct
python mimic/visualize_data.py \
  --robot booster_lower_t1 \
  --npz booster_dataset/imitation_learning/booster_soccer_showdown.npz \
  --fps 30 \
  --episode 0  # Visualize first episode (optional)

#--npz booster_dataset/soccer/booster_t1/goal_kick.npz \
