# encoding=utf-8
import yaml

from chatgpt.pipeline import launch_convert_fs_mp
from chatgpt.pipeline.config import ModelConvertPipelineConfig, PipelineConfig
from chatgpt.utils import logging_initialize


def test():
    
    with open("/cognitive_comp/songzhuoyang/projects/chatgpt/chatgpt/pipeline/scripts/token_level_ppo_lite.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config["pipeline"]["workspace_path"] = "/cognitive_comp/songzhuoyang/workspace/convert/"    
    
    pipeline_config = PipelineConfig()
    pipeline_config.update(config)
    pipeline_config.check()
    convert_pipeline_config = ModelConvertPipelineConfig()
    convert_pipeline_config.update_default_config(pipeline_config)
    convert_pipeline_config.update(config)
    convert_pipeline_config.check()
    logging_initialize(level="debug")
    launch_convert_fs_mp(convert_pipeline_config)
    
    return

if __name__ == "__main__":
    test()