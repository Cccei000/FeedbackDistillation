# encoding=utf-8
import os
import yaml
from chatgpt.pipeline.config import BasicPPOTrainerConfig

script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
example_config_path = {
    "token_level_ppo": os.path.join(script_path, "token_level_ppo_all_param.yml"), 
    "token_level_ppo_naive": os.path.join(script_path, "token_level_ppo_lite.yml")
}

def test():
    
    with open(example_config_path["token_level_ppo"], "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(example_config_path["token_level_ppo_naive"], "r") as f:
        user_config = yaml.load(f, Loader=yaml.FullLoader)
        
    config = BasicPPOTrainerConfig()
    config.update(base_config)
    config.update(user_config)
    
    print(config.__dict__)
    
    
    return


if __name__ == "__main__":
    test()