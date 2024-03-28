from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel,AutoConfig
from chatgpt.nn import RewardModel
from chatgpt.nn.utils import masked_mean

class GPTNeoXRM(RewardModel):
    """
    GPT-NeoX Actor model.

    Args:
        model: Pretrained model
    """

    def __init__(self,
                 model) -> None:
        super().__init__(model)
        

    def forward(self,
                input_ids: torch.LongTensor, 
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.body(input_ids,attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        # find last 1 in attention mask
        if attention_mask[:,0].sum()==attention_mask.shape[0]:
            # right padding
            last_index = attention_mask.sum(dim=1)-1
            last_hidden_states = last_hidden_states[torch.arange(last_hidden_states.shape[0]), last_index]
        else:
            # left padding
            last_hidden_states = last_hidden_states[torch.arange(last_hidden_states.shape[0]), -1]
        values = self.value_head(last_hidden_states).squeeze(-1)# (bs,)
        return values
    
class GPTNeoXRMLit(LightningModule):
    def __init__(self,args, reward_model: GPTNeoXRM):
        super().__init__()
        self.warmup_ratio = args.warmup_ratio
        self.lr = args.rm_lr
        self.reward_model = reward_model
        # self.save_hyperparameters()

    def configure_optimizers(self):
        print(self.trainer.estimated_stepping_batches)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        optimizer = AdamW(self.reward_model.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
    
    def calculate_loss(self,output,pair_length):
        count = 0
        loss = 0
        for l in pair_length:
            output_tmp = output[int(count):int(count+l)]
            count+=l
            output_tmp = torch.combinations(output_tmp, r=2, with_replacement=False)
            loss_func = nn.LogSigmoid()
            loss += torch.mean(-loss_func(output_tmp[:,0]-output_tmp[:,1]))
        return loss
    
    def caculate_compare_acc(self,output,pair_length):
        count = 0
        right = 0
        all_count = 0
        for l in pair_length:
            output_tmp = output[int(count):int(count+l)]
            count+=l
            output_tmp = torch.combinations(output_tmp, r=2, with_replacement=False)
            diff = output_tmp[:,0]-output_tmp[:,1]
            right+=(diff>0).sum()
            all_count+=len(diff)
        acc = right/all_count
        return acc.item(),right,all_count        
    
    def training_step(self, batch, batch_idx):
        rewards = self.reward_model(batch['input_ids'], attention_mask = batch['attention_mask'])
        loss = self.calculate_loss(rewards,batch['pair_length'])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rewards = self.reward_model(batch['input_ids'], attention_mask = batch['attention_mask'])
        loss = self.calculate_loss(rewards,batch['pair_length'])
        acc,right_count,all_count =self.caculate_compare_acc(rewards,batch["pair_length"])
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        rewards = self.reward_model(batch['input_ids'], attention_mask = batch['attention_mask'])
        loss = self.calculate_loss(rewards,batch['pair_length'])
        acc,right_count,all_count =self.caculate_compare_acc(rewards,batch["pair_length"])
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        rewards = self.reward_model(batch['input_ids'], attention_mask = batch['attention_mask'])
        return rewards
    
def modeling_neox_rm(config_path:str, ckpt_path:str, ) -> GPTNeoXRM:

    # 从lightning ckpt加载
    print(f"load from RM Model'{ckpt_path}'...")
    reward_model = GPTNeoXRM(AutoModel.from_config(AutoConfig.from_pretrained(config_path)))
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))['module']
    new_state_dict = {
        key[len("module.reward_model."):]: value for key, value in checkpoint.items()
    }
    reward_model.load_state_dict(state_dict=new_state_dict, strict=True)

    return reward_model