2023.07.25更新

RM训练数据收集脚本：
    
- construct_rm_dataset_elo.sh: 收集/cognitive_comp/songzhuoyang/dialogue_rm_datasets/下的elo排序数据，并和以前的排序数据合并保存到save_base_path

- 其他文件: 用于收集非elo排序数据，目前已经停止标注，可以直接使用construct_rm_dataset_elo.sh的back_up_ds将整理好的非elo数据路径传入（/cognitive_comp/liangyuxin/datasets/RM/0725_backup/merged_dataset）