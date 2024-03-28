# 统一训练框架

* 以[fengshenbang-lm](http://git.team.idea.edu.cn/cognitive-computing/fengshenbang-lm)为基础，本工程用于启动各SFT模型的RLHF训练
* 详细使用方法见[WIKI](http://wiki.team.idea.edu.cn/pages/viewpage.action?pageId=31457577)

## 支持的SFT模型
- [x] GPT-Neox-6B
- [x] Llama-7B
- [x] ChatCLM-6B
- [x] Llama-13B
- [x] Llama2-13B
- [x] Baichuan2-13B

## 支持的RM模型
- [x] Transformer-XL-5B
- [x] Llama-7B
- [x] Llama-13B
- [x] Llama2-13B

## Features
### 训练偏好学习流程
||Token-level RM|Sample-level RM|Token-mix-sample RM| w/o RM|
|---:|:---:|:---:|:---:|:---:|
|Token-level PPO|✅|✅|✅|❌|
|Step-level PPO|✅|✅|✅|❌|
|Sample-level PPO|✅|✅|✅|❌|
|EDPO|❌|❌|❌|✅|

## 流程中支持的生成方式
- [x] Vanila Generation ++
- [x] Best-of-N Generation
- [x] Token-level bfs Generation
- [x] Pipeline Generation



