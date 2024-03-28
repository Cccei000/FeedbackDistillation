# encoding=utf-8
import sys
from argparse import ArgumentParser

import yaml

from chatgpt.pipeline import (check_fs_mp, launch_convert_fs_mp, launch_ppo, launch_edpo,
                              launch_reward_modeling, launch_FD, launch_FD_sandbox)
from chatgpt.pipeline.config import (ModelConvertPipelineConfig,
                                     PipelineConfig, PPOPipelineConfig, EDPOPipelineConfig,
                                     RewardModelingPipelineConfig, FDPipelineConfig)
from chatgpt.utils import logging_initialize, logging_rank_0

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str,
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logging_rank_0(config)
    pipeline_config = PipelineConfig()
    pipeline_config.update(config)
    logging_initialize(level=pipeline_config.logging_level)
    
    if not pipeline_config.check():
        logging_rank_0("Config Error!")
        sys.exit()
    
    logging_rank_0(f"Start Pipeline: {pipeline_config.pipeline}", level="info")
    
    
    if "prepare" in pipeline_config.pipeline or not check_fs_mp(config):
        # 准备：切分模型
        convert_pipeline_config = ModelConvertPipelineConfig()
        convert_pipeline_config.update_default_config(pipeline_config)
        convert_pipeline_config.update(config)
        if not convert_pipeline_config.check():
            sys.exit()
        logging_rank_0("Start Preparing.")
        launch_convert_fs_mp(convert_pipeline_config)
        logging_rank_0("Finish preparing!")

    best_rm_step = None
        
    if "reward_modeling" in pipeline_config.pipeline:
        rm_pipeline_config = RewardModelingPipelineConfig()
        rm_pipeline_config.update_with_default_config(pipeline_config)
        rm_pipeline_config.update(config)
        if not rm_pipeline_config.check():
            sys.exit()
        logging_rank_0("Start Reward Modeling.")
        best_rm_step = launch_reward_modeling(rm_pipeline_config)
        logging_rank_0(f"Finish Reward Modeling. Best step is {best_rm_step}")
        
    
    if "ppo" in pipeline_config.pipeline and "edpo" in pipeline_config.pipeline:
        logging_rank_0("Please select one of the 'ppo' and 'edpo' pipelines.", level="error")
        
    elif "ppo" in pipeline_config.pipeline:
        
        ppo_pipeline_config = PPOPipelineConfig()
        ppo_pipeline_config.update_with_default_config(pipeline_config)
        ppo_pipeline_config.update(config)
        if not ppo_pipeline_config.check():
            sys.exit()
        logging_rank_0("Start PPO Training.")
        launch_ppo(ppo_pipeline_config)
        
        
    elif "edpo" in pipeline_config.pipeline:
        edpo_pipeline_config = EDPOPipelineConfig()
        edpo_pipeline_config.update_with_default_config(pipeline_config)
        edpo_pipeline_config.update(config)
        if not edpo_pipeline_config.check():
            sys.exit()
        logging_rank_0("Start EDPO Training.")
        launch_edpo(edpo_pipeline_config)
    
    elif "FD" in pipeline_config.pipeline:
        FD_pipeline_config = FDPipelineConfig()
        FD_pipeline_config.update_with_default_config(pipeline_config)
        FD_pipeline_config.update(config)
        if not FD_pipeline_config.check():
            sys.exit()
        logging_rank_0("Start FD Training.")
        launch_FD(FD_pipeline_config)
    
    elif "FD_sandbox" in pipeline_config.pipeline:
        FD_pipeline_config = FDPipelineConfig()
        FD_pipeline_config.update_with_default_config(pipeline_config)
        FD_pipeline_config.update(config)
        if not FD_pipeline_config.check():
            sys.exit()
        logging_rank_0("Start FD_sandbox Training.")
        launch_FD_sandbox(FD_pipeline_config)

    logging_rank_0("Finished!")
    
        
        
