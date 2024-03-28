# encoding=utf-8
from .deepspeed_strategy import (add_megatron_deepspeed_args,
                                 build_deepspeed_config,
                                 get_save_checkpoint_callback,
                                 initialize_megatron, setup_inference_engine,
                                 setup_model_and_optimizer)


__all__ = ["add_megatron_deepspeed_args", "initialize_megatron", "build_deepspeed_config", "setup_model_and_optimizer", "get_save_checkpoint_callback", "setup_inference_engine"]