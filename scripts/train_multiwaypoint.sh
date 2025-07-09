#!/bin/sh
env="MultiWaypoint"
scenario="1/multiwaypoint"
algo="ppo"
exp="v1"
seed=6

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 2 --n-rollout-threads 8 --cuda \
    --log-interval 1 --save-interval 10 \
    --num-mini-batch 4 --buffer-size 3000 --num-env-steps 1e7 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 1 --entropy-coef 0.001 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --user-name "adrien-maugueret" --wandb-name "waypoint-jsbsim" --use-wandb \ 
    #--render-mode "real_time" --use-eval True --eval-interval 1
