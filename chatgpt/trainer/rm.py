import gc
import os
from abc import ABC
from collections import defaultdict
from typing import Optional

# import loralib as lora
import torch
import torch.nn as nn
import wandb
from datasets import Dataset
# from torch.utils.data import DataLoader
from tqdm import tqdm

# from chatgpt.dataset import RewardDataset
# from chatgpt.nn import PairWiseLoss
from chatgpt.utils import print_rank_0
from chatgpt.nn.utils import masked_mean
from chatgpt.utils import is_rank_0, logging_rank_0
import torch.distributed as dist
from fengshen_inner.models.megatron import mpu

def masked_mean_with_eos(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    eos_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in mask], dtype=torch.int64)
    sample_reward = tensor[torch.arange(tensor.shape[0]), eos_index]
    tensor[torch.arange(tensor.shape[0]), eos_index]=0.
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    mean = (mean+sample_reward)/2.0
    return mean

def mix_reward(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    eos_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in mask], dtype=torch.int64)
    sample_reward = tensor[torch.arange(tensor.shape[0]), eos_index]
    tensor[torch.arange(tensor.shape[0]), eos_index]=0.
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = tensor * mask
    tensor_sum = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor_sum / (mask_sum + 1e-8)
    return mean,sample_reward,tensor


class RMTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        optim,
        lr_scheduler,
        setup_dataloader_func,
        ckpt_saving_func,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        max_epochs: int = 2,
        logger = None,       
        collate_fn=None,
        val_check_interval=0.05,
        l2_beta=0.01,
        predict_dataset=None,
        test_dataset=None,
        save_best_n_ckpt:int=0,
        ckpt_deleting_func:Optional[callable]=None,
        special_test_dataset=None, 
        granularity="sample", # sample-level or token-level or token-sample-mix reward
    ) -> None:
        """_summary_

        Args:
            model (_type_): rm
            optim (_type_): optimizer
            lr_scheduler (_type_): lr_scheduler
            setup_dataloader_func (_type_): _description_
            ckpt_saving_func (_type_): _description_
            train_dataset (Dataset): _description_
            eval_dataset (Dataset): _description_
            max_epochs (int, optional): _description_. Defaults to 2.
            logger (_type_, optional): _description_. Defaults to None.
            collate_fn (_type_, optional): _description_. Defaults to None.
            val_check_interval (float, optional): _description_. Defaults to 0.05.
            l2_beta (float, optional): _description_. Defaults to 0.01.
            predict_dataset (_type_, optional): _description_. Defaults to None.
            test_dataset (_type_, optional): _description_. Defaults to None.
            save_best_n_ckpt (int, optional): save_best_n_ckpt. Defaults to 0, meaning saving all ckpt
        """
        super().__init__()
        self.model = model
        self.optimizer = optim
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        self.logger = logger
        self.l2_beta=l2_beta
        self.train_steps = 0
        self.steps = 0
        self.ckpt_saving_func = ckpt_saving_func
        self.save_best_n_ckpt = save_best_n_ckpt
        self.ckpt_deleting_func = ckpt_deleting_func
        self.granularity = granularity
        self.is_training = True
        self.set_dataloaders(train_dataset,eval_dataset,test_dataset,predict_dataset,special_test_dataset,setup_dataloader_func,collate_fn)
        self.val_check_interval = int(val_check_interval*len(self.train_dataloader))
        logging_rank_0(f"Reward Modeling: val_check_interval={self.val_check_interval} | len(train_dataloader)={len(self.train_dataloader)}")

    def fit(self,) -> int:
        pbar_epoch = tqdm(range(self.max_epochs), desc="Epoch", leave=False, position=0, disable=not is_rank_0())
        val_results = []    # 记录每次验证结果
        for epoch in pbar_epoch:
            self._on_epoch_start()
            pbar_batch = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Batch", leave=False, position=1, disable=not is_rank_0())
            for i,batch in pbar_batch:
                if i%self.val_check_interval==0:# and i!=0:
                    self.is_training = False
                    val_acc,val_loss = self.validation_loop()    
                    pbar_batch.set_postfix_str(f"Val Result: val_acc={round(val_acc.item(), 4)} | val_loss={round(val_loss.item(), 4)}")

                    # 检查历史ckpt的验证效果，替代较差的ckpt
                    self.ckpt_saving_func(self.global_steps, self.model)
                    val_results.append((val_acc.item(), -val_loss.item(), self.steps))
                    print("val_results",val_results)
                    if self.save_best_n_ckpt <= 0 or len(val_results) <= self.save_best_n_ckpt or self.ckpt_deleting_func is None:
                        # self.ckpt_saving_func(self.steps, self.model)
                        # do nothing
                        pass
                    else:
                        val_results = sorted(val_results)
                        # if val_results[0][2] != self.steps:
                        #     self.ckpt_deleting_func(val_results[0][2])
                        #     self.ckpt_saving_func(self.steps, self.model)
                        val_results = val_results[1:]
                        
                    gc.collect()
                    torch.cuda.empty_cache()
                        
                    # if self.special_test_dataloader is not None:
                    #     special_test_acc = self.special_test_loop()
                    #     gc.collect()
                    #     torch.cuda.empty_cache()
                    self.is_training=True
                loss = self.training_step(batch)
                self.logger.log_metrics({"training_loss":loss.item()},step=self.global_steps,metrics_group="train")
                self.logger.log_metrics({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_steps, metrics_group='lr')
                self.steps+=1
                gc.collect()
                torch.cuda.empty_cache()   
        val_results = sorted(val_results)  
        return val_results[-1][2]

    def training_step(self, batch):
        self.model.train()
        device = torch.cuda.current_device()
        rewards = self.model(batch['input_ids'].to(device), attention_mask = batch['attention_mask'].to(device),action_mask=batch['action_mask'].to(device))
        if self.granularity=="sample":
            loss,_ = self.calculate_loss_sample(rewards,batch['pairs'].to(device),action_mask=batch['action_mask'].to(device))
        elif self.granularity=="token":
            loss,_ = self.calculate_loss_token(rewards,batch['pairs'].to(device),action_mask=batch['action_mask'].to(device))
        elif self.granularity == "token_mix_sample":
            loss,_ = self.calculate_loss_token_mix_sample(rewards,batch['pairs'].to(device),action_mask=batch['action_mask'].to(device))
        else:
            raise Exception("granularity not supported")

        self.model.backward(loss)
        self.model.step()
        self.model.zero_grad()
        return loss    
    
    def validation_loop(self):
        self.model.eval()
        device = torch.cuda.current_device()
        losses = []
        tasks = []
        pos_rewards,neg_rewards = [],[]
        pos_len,neg_len = [],[]
        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                action_mask = batch['action_mask'].to(device)
                _pairs = batch['pairs'].to(device)
                rewards = self.model(batch['input_ids'].to(device), attention_mask = batch['attention_mask'].to(device), action_mask = action_mask)
                if self.granularity=="sample":
                    _loss, _sample_rewards = self.calculate_loss_sample(rewards,_pairs,action_mask=action_mask)
                elif self.granularity=="token":
                    _loss, _sample_rewards = self.calculate_loss_token(rewards,_pairs,action_mask=action_mask)
                elif self.granularity == "token_mix_sample":
                    _loss, _sample_rewards = self.calculate_loss_token_mix_sample(rewards,_pairs,action_mask=action_mask)
                else:
                    raise Exception("granularity not supported")

                losses.append(_loss)
                pos_rewards.append(_sample_rewards.take(_pairs[:, 0]))
                neg_rewards.append(_sample_rewards.take(_pairs[:, 1]))
                tasks.extend([batch['task'][k] for k in _pairs[:, 0]])
                action_len = action_mask.sum(dim=1)
                pos_len.append(action_len.take(_pairs[:, 0]))
                neg_len.append(action_len.take(_pairs[:, 1]))

        val_acc,val_loss = self.validation_log(losses,pos_rewards,neg_rewards,pos_len,neg_len,tasks)

        return val_acc,val_loss

    def validation_log(self,losses,pos_rewards,neg_rewards,pos_len,neg_len,tasks):
        loss = torch.stack(losses)
        loss = loss.mean()
        dist.all_reduce(loss,dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
        loss = loss/mpu.get_data_parallel_world_size()

        pos_rewards = torch.cat(pos_rewards)
        neg_rewards = torch.cat(neg_rewards)

        reward_diffs=pos_rewards-neg_rewards
        right=(reward_diffs>0).sum()
        all_count=torch.tensor([len(reward_diffs)],device=reward_diffs.device)
        dist.all_reduce(right,dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
        dist.all_reduce(all_count,dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
        
        val_acc=right/all_count.item()

        self.logger.log_metrics({
            "validation_loss":loss.item(),
            "validation_acc":val_acc,
            },step=self.global_steps,metrics_group="validation")

        self.logger.log_metrics({
            "pos_reward": wandb.Histogram(pos_rewards.float().cpu()),
            "neg_reward":wandb.Histogram(neg_rewards.float().cpu()),
            "reward_diff":wandb.Histogram(reward_diffs.float().cpu()),
            },step=self.global_steps,metrics_group="validation-detail") 

        pos_len = torch.cat(pos_len)
        neg_len = torch.cat(neg_len)
        len_diffs = pos_len-neg_len
        pos_diff = reward_diffs.where(len_diffs>0,torch.Tensor([-1.0]).to(len_diffs.device))
        neg_diff = reward_diffs.where(len_diffs<0,torch.Tensor([-1.0]).to(len_diffs.device))
        val_acc_pos = (pos_diff>0).sum()/(len_diffs>0).sum()
        val_acc_neg = (neg_diff>0).sum()/(len_diffs<0).sum()
        self.logger.log_metrics({
            "validation_acc_pos":val_acc_pos,
            "validation_acc_neg":val_acc_neg,
            },step=self.global_steps,metrics_group="validation-detail")
        
        task_acc = defaultdict(list)
        for task, reward_diff in zip(tasks,reward_diffs):
            task_acc[task].append(reward_diff>0)
        #TODO proper all_reduce
        task_acc = {k:sum(v)/len(v) for k,v in task_acc.items() if len(v)>200}
        print("task_acc",task_acc)
        self.logger.log_metrics(task_acc,step=self.global_steps,metrics_group="validation-detail")

        return val_acc,loss

    def calculate_loss_sample(self,rewards,pairs,action_mask):
        loss = self.calculate_loss(sample_rewards=rewards,rewards=rewards, pairs=pairs)
        return loss,rewards
    
    def calculate_loss_token(self,rewards,pairs,action_mask):
        sample_rewards = masked_mean(rewards,action_mask)
        loss = self.calculate_loss(sample_rewards=sample_rewards,rewards=rewards, pairs=pairs)
        return loss,sample_rewards
    
    def calculate_loss_token_mix_sample(self,rewards,pairs,action_mask):
        sample_rewards_of_token_level,sample_rewards,rewards_of_token_level = mix_reward(rewards,action_mask)
        sample_loss = self.calculate_loss(sample_rewards=sample_rewards,rewards=sample_rewards, pairs=pairs)
        token_loss = self.calculate_loss(sample_rewards=sample_rewards_of_token_level,rewards=rewards_of_token_level, pairs=pairs)
        loss = (sample_loss+token_loss)/2.0
        sample_rewards = (sample_rewards_of_token_level+sample_rewards)/2.0
        return loss,sample_rewards

    def calculate_loss(self,sample_rewards,rewards,pairs):
        # sample_rewards：样本级reward
        # rewards：RM输出reward（可能是样本级或token级）
        loss_func = nn.Sigmoid()
        pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
        pos_rewards = sample_rewards.take(pos_ids)
        neg_rewards = sample_rewards.take(neg_ids)
        pos = rewards[pos_ids]
        neg = rewards[neg_ids]
        l2 = self.l2_beta * 0.5 * (pos**2+neg**2).mean()
        pair_loss = -torch.log(loss_func(pos_rewards - neg_rewards)).mean()
        loss = pair_loss +  l2
        return loss

    def set_dataloaders(self,train_dataset,eval_dataset,test_dataset,predict_dataset,special_test_dataset,setup_dataloader_func,collate_fn):
        self.train_dataloader = setup_dataloader_func(train_dataset, collate_fn=collate_fn['train'])
        self.eval_dataloader = setup_dataloader_func(eval_dataset, collate_fn=collate_fn['eval'])
        self.test_dataloader = setup_dataloader_func(test_dataset, collate_fn=collate_fn['test'])  if test_dataset is not None else None
        self.predict_dataloader = setup_dataloader_func(predict_dataset, collate_fn=collate_fn['predict']) if predict_dataset is not None else None
        self.special_test_dataloader = setup_dataloader_func(special_test_dataset, collate_fn=collate_fn['special_test']) if special_test_dataset is not None else None

    def on_train_batch_start(self):
        pass

    def _on_epoch_start(self):
        pass
    
    @property
    def global_steps(self) -> int:
        return self.train_steps + self.steps 

