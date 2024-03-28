# encoding=utf-8
from chatgpt.utils import logging_initialize
from chatgpt.pipeline.utils import load_jsonline_data, concat_prompt


if __name__ == "__main__":
    
    path = "/cognitive_comp/songzhuoyang/processed_data/pipeline_dev_dataset.jsonl"
    logging_initialize(level="debug")
    dataset = load_jsonline_data(
        path
    )
    dataset = load_jsonline_data(
        path,
        {"model": "<bot>:"}
    )
    dataset = load_jsonline_data(
        path,
        {"model": "<bot>:", "human": "<human>:"}
    )
    dataset = load_jsonline_data(
        path,
        {"model": "<bot>:", "human": "<human>:"},
        "\n"
    )
    print(dataset)
    dataset = concat_prompt(dataset, {"model": "<bot>:", "human": "<human>:"}, "\n")
    print(dataset)
    
    
    
    