import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
sys.setrecursionlimit(10000)
import argparse

from pathlib import Path
import configue

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.gpu_stats import print_gpu_utilization


def main(config_file: Path) -> None:
    print_gpu_utilization()
    print("Loading config")
    config = configue.load(config_file, sub_path="config")
    print("Creating Setup")
    if isinstance(config, ColModelTrainingConfig):
        app = ColModelTraining(config)
    else:
        raise ValueError("Config must be of type ColModelTrainingConfig")

    if config.run_train:
        print("Training model")
        app.train()
        app.save(config_file=config_file)
    if config.run_eval:
        if app.model.device.type=='cpu':
            print("attach model to cuda")
            app.model.to("cuda")
        print("Running evaluation")
        app.eval()
    print("Done!")


if __name__ == "__main__":
    # config_file = "./scripts/configs/qwen2/inference_post_proj_flatten.yaml"
    # # config_file = "./scripts/configs/qwen2/inference_pre_proj_flatten.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_flatten.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_selected.yaml"
    # config_file = "./scripts/configs/qwen2/debug_infer.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_flatten_well-init.yaml"
    # config_file = "./scripts/configs/idefics/train_colidefics2_model.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_channel-pool.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_2dpool.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_local.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_2dpool_local.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_well-init_local.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_flatten_high.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_prune.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_celoss.yaml"
    # config_file = "./scripts/configs/qwen2/train_biqwen2_model.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_cluster.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_pre_llm_cluster.yaml"
    # config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_pre_llm_2dpool.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=Path, default="./scripts/configs/qwen2/train_colqwen2_model.yaml")
    args = parser.parse_args()

    main(args.config_file)
