#CUDA_VISIBLE_DEVICES=0 python medclip_finetuner.py --gpus 1 --dataset rsna --data_pct 0.01
#CUDA_VISIBLE_DEVICES=0 python medclip_finetuner.py --gpus 1 --dataset rsna --data_pct 0.1
#CUDA_VISIBLE_DEVICES=0 python medclip_finetuner.py --gpus 1 --dataset rsna --data_pct 1


CUDA_VISIBLE_DEVICES=0 python medclip_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.01
CUDA_VISIBLE_DEVICES=0 python medclip_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.1
CUDA_VISIBLE_DEVICES=0 python medclip_finetuner.py --gpus 1 --dataset chexpert --data_pct 1 --batch_size 96
