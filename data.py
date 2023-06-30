import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class EncodedAssistQA(Dataset):
    def __init__(self, cfg, is_train):
        super().__init__()
        # for loveu@cvpr2022. paper is updated for this.
        root = cfg.DATASET.TRAIN if is_train else cfg.DATASET.VAL
        root_reuse = cfg.DATASET.TRAIN # For para, hands, video, global, local (except QAs)
        root_hand = cfg.INPUT.HANDPATH
        samples = []
        self.dual = cfg.INPUT.DUAL
        for t in os.listdir(root):
            sample = torch.load(os.path.join(root, t, cfg.INPUT.QA), map_location="cpu")
            for s in sample:
                s["video"] = os.path.join(root_reuse, t, cfg.INPUT.VIDEO)
                if self.dual:
                    s["ego"] = os.path.join(root_reuse, t, cfg.INPUT.EGOVLP)
                s["script"] = os.path.join(root_reuse, t, cfg.INPUT.SCRIPT)
                s["para"] = os.path.join(root_reuse, t, cfg.INPUT.PARA)
                s["hand"] = os.path.join(root_hand, t, cfg.INPUT.HAND)
            samples.extend(sample)
        self.samples = samples
        
    def __getitem__(self, index):
        sample = self.samples[index]
        video = torch.load(sample["video"], map_location="cpu")

        timestamp_script = torch.load(sample["script"], map_location="cpu")
        sents_timestamp, script = timestamp_script

        timestamp_para = torch.load(sample['para'], map_location="cpu")
        paras_timestamp, function_para = timestamp_para
        
        score_hand = torch.load(sample["hand"], map_location="cpu")

        question = sample["question"]
        actions = sample["answers"]
        meta = {
            'question': sample['src_question'], 'folder': sample['folder'], 
            'paras_score': sample['paras_score'], 'paras_timestamp': paras_timestamp, 
            'sents_score': sample['sents_score'], 'sents_timestamp': sents_timestamp,
            'hand_score': score_hand
        }
        if 'correct' in sample:
            label = torch.tensor(sample['correct']) - 1 
            # NOTE here, start from 1
        else:
            label = None
        # return video, script, question, function_para, actions, label, meta
        
        if self.dual:
            ego_video = torch.load(sample["ego"], map_location="cpu")
            return video, ego_video, script, question, function_para, actions, label, meta
        else:
            return video, script, question, function_para, actions, label, meta

        
    def __len__(self, ):
        return len(self.samples)

    @staticmethod
    def collate_fn(samples):
        return samples

class EncodedAssistQADataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def train_dataloader(self): 
        cfg = self.cfg
        trainset = EncodedAssistQA(cfg, is_train=True)
        return DataLoader(trainset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=True, drop_last=True, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    def val_dataloader(self):
        cfg = self.cfg
        valset = EncodedAssistQA(cfg, is_train=False)
        return DataLoader(valset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=False, drop_last=False, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    
def build_data(cfg):
    return EncodedAssistQADataModule(cfg)