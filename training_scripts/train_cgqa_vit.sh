#!/bin/bash

#seed=42
#for seed in 42 40 44
#do
#python -m torch.distributed.launch \
#        --nproc_per_node=1 \
#        --use_env main.py \
#        cgqa_hideprompt_5e \
#        --original_model vit_base_patch16_224 \
#        --model vit_base_patch16_224 \
#        --batch-size 24 \
#        --data-path ../datasets \
#        --output_dir ./output/cgqa_sup21k_multi_centroid1_mlp_2_seed$seed \
#        --epochs 20 \
#        --sched constant \
#        --seed $seed \
#        --train_inference_task_only \
#        --num_tasks 10 \
#        --n_centroids 1 \
#        --lr 0.0005
#done

for seed in 40 44
do
python -m torch.distributed.launch \
	--nproc_per_node=1 \
	--master_port='29501' \
	--use_env main.py \
	cgqa_hideprompt_5e \
	--model vit_base_patch16_224 \
	--original_model vit_base_patch16_224 \
	--batch-size 24 \
	--epochs 50 \
	--data-path ../datasets \
	--ca_lr 0.005 \
	--crct_epochs 30 \
	--seed $seed \
	--prompt_momentum 0.01 \
	--reg 0.1 \
	--length 5 \
	--sched step \
	--larger_prompt_lr \
	--trained_original_model ./output/cgqa_sup21k_multi_centroid1_mlp_2_seed$seed \
  --num_tasks 10 \
  --n_centroids 1 \
	--output_dir ./output/cgqa_vit_pe_centroid1_seed$seed > ./output/cgqa_vit_pe_centroid1_seed${seed}.out 2>&1
done
