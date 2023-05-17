#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p normal
#S#BATCH -p normal
#SSBATCH -p use-everything
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

python main.py --dataset wm --model_name ewc --init_cls 1 --device 0 --convnet_type wm -incre 1
