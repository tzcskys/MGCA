CUDA_VISIBLE_DEVICES=0 python gloria_segmenter.py --gpus 1 --data_pct 0.01 --dataset siim --learning_rate 5e-4 --batch_size 8 --seed 66
CUDA_VISIBLE_DEVICES=1 python gloria_segmenter.py --gpus 1 --data_pct 0.1 --dataset siim --learning_rate 5e-4 --seed 66
CUDA_VISIBLE_DEVICES=1 python gloria_segmenter.py --gpus 1 --data_pct 1 --dataset siim --learning_rate 5e-4 --seed 66

#CUDA_VISIBLE_DEVICES=1 python gloria_segmenter.py --gpus 1 --data_pct 1 --dataset siim --learning_rate 5e-4 --batch_size 16 --seed 66


#CUDA_VISIBLE_DEVICES=0 python gloria_segmenter.py --gpus 1 --data_pct 0.01 --dataset rsna --batch_size 16 --learning_rate 5e-4
#CUDA_VISIBLE_DEVICES=0 python gloria_segmenter.py --gpus 1 --data_pct 0.1 --dataset rsna --batch_size 16 --learning_rate 5e-4
#CUDA_VISIBLE_DEVICES=0 python gloria_segmenter.py --gpus 1 --data_pct 1 --dataset rsna --batch_size 16 --learning_rate 5e-4