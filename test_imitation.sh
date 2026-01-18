

python imitation_learning/test.py \
  --restore_path exp/booster/Debug/LowerT1GoaliePenaltyKick_20260115-091704_bc \
  --restore_epoch 1000000 \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \
  --eval_episodes 5 \