python imitation_learning/train.py \
  --agents bc_mse \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \


python imitation_learning/train.py \
  --agents gcbc_mse \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \

python imitation_learning/train.py \
  --agents gcbc \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \

python imitation_learning/train.py \
  --agents gciql \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \

python imitation_learning/train.py \
  --agents iql \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \

python imitation_learning/train.py \
  --agents hiql \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \

python imitation_learning/train.py \
  --agents iql \
  --dataset_dir booster_dataset/imitation_learning/booster_soccer_showdown.npz \
# 'gcbc', 'gcbc_mse', 'gciql', 'hiql', 'bc', 'bc_mse', 'iql'
