# export MODEL_PATH=/mnt/petrelfs/mayubo/colpali/models_new/colqwen2_post_proj_cluster_factor9_0210_revised/checkpoint-2310
export MODEL_PATH=/mnt/petrelfs/mayubo/colpali/models_new/colqwen2_post_proj_flatten_factor9
# python ./identify_valid_patch.py --dataset_name_out "docvqa" --prune_criteron cluster-score --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "arxivqa" --prune_criteron cluster-score --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "infovqa" --prune_criteron cluster-score --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "tabfquad" --prune_criteron cluster-score --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "tatdqa" --prune_criteron cluster-score --peft_name $MODEL_PATH
python ./identify_valid_patch.py --dataset_name_out "shift" --prune_criteron cluster-score --peft_name $MODEL_PATH


# python ./identify_valid_patch.py --dataset_name_out "docvqa" --prune_criteron cluster-random --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "arxivqa" --prune_criteron cluster-random --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "infovqa" --prune_criteron cluster-random --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "tabfquad" --prune_criteron cluster-random --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "tatdqa" --prune_criteron cluster-random --peft_name $MODEL_PATH
# python ./identify_valid_patch.py --dataset_name_out "shift" --prune_criteron cluster-random --peft_name $MODEL_PATH