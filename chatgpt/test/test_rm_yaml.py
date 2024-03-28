# encoding=utf-8
import sys
import json

import yaml

from chatgpt.pipeline.config import (PipelineConfig,
                                     RewardModelingPipelineConfig)
from chatgpt.utils import logging_initialize, logging_rank_0


def test():
    yml_path = "/cognitive_comp/songzhuoyang/projects/chatgpt/chatgpt/pipeline/scripts/token_level_full_lite.yml"
    with open(yml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    pipeline_config = PipelineConfig()
    pipeline_config.update(config)
    
    if not pipeline_config.check():
        sys.exit()
    
    logging_initialize(level=pipeline_config.logging_level)
    logging_rank_0(f"Start Pipeline: {pipeline_config.pipeline}", level="info")
    
    rm_pipeline_config = RewardModelingPipelineConfig()
    rm_pipeline_config.update_with_default_config(pipeline_config)
    rm_pipeline_config.update(config)
    if not rm_pipeline_config.check():
        sys.exit()
    
    with open("/cognitive_comp/songzhuoyang/workspace/chatgpt/token_level_rm.json", "w+") as f:
        f.write(json.dumps(rm_pipeline_config.__dict__, ensure_ascii=False))
    
    
    return


if __name__ == "__main__":
    test()