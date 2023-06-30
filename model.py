import torch, os, json
from torch import nn
from torch.functional import F
from pytorch_lightning import LightningModule


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 4*in_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4*in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class MLP1(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Q2A_Function_blip(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP1(cfg.INPUT.VISUAL_DIM,  cfg.INPUT.VISUAL_OUT)
        self.mlp_t = MLP1(cfg.INPUT.TEXT_DIM, cfg.INPUT.TEXT_OUT)
        self.mlp_pre = MLP1(cfg.INPUT.VISUAL_OUT * 2 + cfg.INPUT.TEXT_OUT * 2, cfg.MODEL.DIM_STATE) # 1024, 768 #1024 256?
        
        self.state = torch.randn(cfg.MODEL.DIM_STATE, device="cuda")
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP1(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        
        self.function_centric = cfg.MODEL.FUNCTION_CENTRIC
        self.cfg = cfg
        self.egovlp = cfg.INPUT.ISEGOVLP

    def forward(self, batch):
        loss, count = 0, 0
        results = []
        for video, script, question, para, actions, label, meta in batch:
            # for text
            if self.function_centric:
                score = torch.tensor(meta['paras_score']).softmax(dim=0).cuda() # 1 * n
                timestamps = meta['paras_timestamp']
                para = self.mlp_t(para) # n, 768
                para = torch.matmul(score, para) # weight each paragraph
            else:
                score = torch.tensor(meta['sents_score']).softmax(dim=0).cuda()
                timestamps = meta['sents_timestamp']
                script = self.mlp_t(script)
                script = torch.matmul(score, script)
            text_seg = para if self.function_centric else script

            # for visual
            hand_score = meta["hand_score"]
            video = self.mlp_v(video) # video shape? 
            video_seg = []
            for seg in timestamps:
                if self.egovlp:
                    video_seg = video
                    break
                else:
                    if seg[0] >= seg[1]:
                        video_seg.append(video[seg[0]])
                    else:
                        video_seg.append(video[seg[0]:seg[1]].mean(dim=0))
            if not self.egovlp:
                video_seg = torch.stack(video_seg) # n, 768
            video_seg = torch.matmul(score, video_seg) # 768 (weight combination)
                
            question = self.mlp_t(question)
            
            state = self.state
            scores = []
            for i, actions_per_step in enumerate(actions): # in actions, all the actions in the all steps are stored. such as 3 steps * 14 actions
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step]) # action: dic{text: ..., button: ...}
                # a_texts: Num_actions (A), 768
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_v(
                    torch.stack(a_buttons).view(A, -1, 1024) 
                ).view(A, -1) # A, 768
                qa = question + a_texts # A, 768

                inputs = torch.cat(
                    [video_seg.expand(A, -1),  text_seg.expand(A, -1), qa.view(A, -1), a_buttons.view(A, -1)],
                    dim=1
                )
                inputs = self.mlp_pre(inputs) # A, 768
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs)) # inputs, hidden_states
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                # states A, 768
                logits = self.proj(states) # logits 14, 1
                if self.training:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training: # train
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training):
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                results.append(meta)
        if self.training:
            return loss / count
        else:
            return results


class Q2A_Interaction_Global(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP(cfg.INPUT.VISUAL_DIM, cfg.INPUT.VISUAL_OUT)
        self.mlp_t = MLP(cfg.INPUT.TEXT_DIM, cfg.INPUT.TEXT_OUT)
        self.mlp_pre = MLP(cfg.INPUT.VISUAL_OUT * 7,  cfg.MODEL.DIM_STATE)
        
        self.state = torch.randn(cfg.MODEL.DIM_STATE, device="cuda")
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        
        self.function_centric = cfg.MODEL.FUNCTION_CENTRIC
        self.cfg = cfg
        self.egovlp = cfg.INPUT.ISEGOVLP
        # if self.egovlp:
        self.mlp_vit = MLP1(1024, 512)

    def forward(self, batch):
        loss, count = 0, 0
        results = []
        for video, video_global,  script, question, para, actions, label, meta in batch:
            # for text
            if self.function_centric:
                score = torch.tensor(meta['paras_score']).softmax(dim=0).cuda() # 1 * n
                timestamps = meta['paras_timestamp']
                para = self.mlp_t(para) # n, 768
                para = torch.matmul(score, para) # weight each paragraph
            else:
                score = torch.tensor(meta['sents_score']).softmax(dim=0).cuda()
                timestamps = meta['sents_timestamp']
                script = self.mlp_t(script)
                script = torch.matmul(score, script)
            text_seg = para if self.function_centric else script

            # for visual
            hand_score = meta["hand_score"]
            video = self.mlp_vit(video) # video shape? 
            video_global = self.mlp_v(video_global)
            video_seg = []
            video_seg_global = video_global
            for seg in timestamps:
                if seg[0] >= seg[1]:
                    video_seg.append(video[seg[0]])
                else:
                    video_seg.append(torch.matmul(hand_score[seg[0]:seg[1]].softmax(dim=0), video[seg[0]:seg[1]]))  # T * D (T is the corresponding function timestamps)
            video_seg = torch.stack(video_seg) # n, 768
            video_seg = torch.matmul(score, video_seg) # 768 (weight combination)
            video_seg_global = torch.matmul(score, video_seg_global)    
            question = self.mlp_t(question)
            
            state = self.state
            scores = []
            for i, actions_per_step in enumerate(actions): # in actions, all the actions in the all steps are stored. such as 3 steps * 14 actions
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step]) # action: dic{text: ..., button: ...}
                # a_texts: Num_actions (A), 768
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_vit(
                    # torch.stack(a_buttons).view(A, -1, a_texts.shape[1]) 
                    torch.stack(a_buttons).view(A, -1, 1024) 
                ).view(A, -1) # A, 768
                qa = question + a_texts # A, 768

                inputs = torch.cat(
                    [video_seg.expand(A, -1), video_seg_global.expand(A, -1),  text_seg.expand(A, -1), qa.view(A, -1), a_buttons.view(A, -1)],
                    dim=1
                )
                inputs = self.mlp_pre(inputs) # A, 768
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs)) # inputs, hidden_states
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                # states A, 768
                logits = self.proj(states) # logits 14, 1
                if self.training:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training: # train
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training):
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                results.append(meta)
        if self.training:
            return loss / count
        else:
            return results
        

class Q2A_Interaction_clip(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP1(cfg.INPUT.VISUAL_DIM, cfg.INPUT.VISUAL_OUT)
        self.mlp_t = MLP1(cfg.INPUT.TEXT_DIM, cfg.INPUT.TEXT_OUT)
        self.mlp_pre = MLP1(cfg.INPUT.VISUAL_OUT * 2 + cfg.INPUT.TEXT_OUT *2, cfg.MODEL.DIM_STATE) # 1024, 768 #1024 256?
        
        self.state = torch.randn(cfg.MODEL.DIM_STATE, device="cuda")
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP1(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        
        self.function_centric = cfg.MODEL.FUNCTION_CENTRIC
        self.cfg = cfg
        self.egovlp = cfg.INPUT.ISEGOVLP

    def forward(self, batch):
        loss, count = 0, 0
        results = []
        for video, script, question, para, actions, label, meta in batch:
            # for text
            if self.function_centric:
                score = torch.tensor(meta['paras_score']).softmax(dim=0).cuda() # 1 * n
                timestamps = meta['paras_timestamp']
                para = self.mlp_t(para) # n, 768
                para = torch.matmul(score, para) # weight each paragraph
            else:
                score = torch.tensor(meta['sents_score']).softmax(dim=0).cuda()
                timestamps = meta['sents_timestamp']
                script = self.mlp_t(script)
                script = torch.matmul(score, script)
            text_seg = para if self.function_centric else script

            hand_score = meta["hand_score"]
            video = self.mlp_v(video) # video shape? 
            video_seg = []
            for seg in timestamps:
                if self.egovlp:
                    video_seg = video
                    break
                else:
                    if seg[0] >= seg[1]:
                        video_seg.append(video[seg[0]])
                    else:
                        video_seg.append(torch.matmul(hand_score[seg[0]:seg[1]].softmax(dim=0), video[seg[0]:seg[1]]))
            if not self.egovlp:
                video_seg = torch.stack(video_seg) # n, 768
            video_seg = torch.matmul(score, video_seg) # 768 (weight combination)
                
            question = self.mlp_t(question)
            
            state = self.state
            scores = []
            for i, actions_per_step in enumerate(actions): # in actions, all the actions in the all steps are stored. such as 3 steps * 14 actions
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step]) # action: dic{text: ..., button: ...}
                # a_texts: Num_actions (A), 768
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_v(
                    torch.stack(a_buttons).view(A, -1, 1024) 
                ).view(A, -1) # A, 768
                qa = question + a_texts # A, 768

                inputs = torch.cat(
                    [video_seg.expand(A, -1),  text_seg.expand(A, -1), qa.view(A, -1), a_buttons.view(A, -1)],
                    dim=1
                )
                inputs = self.mlp_pre(inputs) # A, 768
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs)) # inputs, hidden_states
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                # states A, 768
                logits = self.proj(states) # logits 14, 1
                if self.training:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training: # train
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training):
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                results.append(meta)
        if self.training:
            return loss / count
        else:
            return results


class Q2A_Interaction_Local(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.mlp_v = MLP(cfg.INPUT.VISUAL_DIM, cfg.INPUT.VISUAL_OUT)
        self.mlp_t = MLP(cfg.INPUT.TEXT_DIM, cfg.INPUT.TEXT_OUT)
        self.mlp_pre = MLP(cfg.INPUT.VISUAL_DIM * 4, cfg.MODEL.DIM_STATE) # 1024, 768 #1024 256?
        
        self.state = torch.randn(cfg.MODEL.DIM_STATE, device="cuda")
        if cfg.MODEL.HISTORY.ARCH == "mlp":
            self.proj = MLP(cfg.MODEL.DIM_STATE*2, 1)
        elif cfg.MODEL.HISTORY.ARCH == "gru":
            self.gru = nn.GRUCell(cfg.MODEL.DIM_STATE, cfg.MODEL.DIM_STATE)
            self.proj = MLP(cfg.MODEL.DIM_STATE, 1)
        else:
            assert False, "unknown arch"
        
        self.history_train = cfg.MODEL.HISTORY.TRAIN
        self.history_val = cfg.MODEL.HISTORY.VAL
        
        self.function_centric = cfg.MODEL.FUNCTION_CENTRIC
        self.cfg = cfg
        self.egovlp = cfg.INPUT.ISEGOVLP

    def forward(self, batch):
        loss, count = 0, 0
        results = []
        for video, script, question, para, actions, label, meta in batch:
            # for text
            if self.function_centric:
                score = torch.tensor(meta['paras_score']).softmax(dim=0).cuda() # 1 * n
                timestamps = meta['paras_timestamp']
                para = self.mlp_t(para) # n, 768
                para = torch.matmul(score, para) # weight each paragraph
            else:
                score = torch.tensor(meta['sents_score']).softmax(dim=0).cuda()
                timestamps = meta['sents_timestamp']
                script = self.mlp_t(script)
                script = torch.matmul(score, script)
            text_seg = para if self.function_centric else script

            # for visual
            hand_score = meta["hand_score"]
            video = self.mlp_v(video) # video shape? 
            video_seg = []
            for seg in timestamps:
                if self.egovlp:
                    video_seg = video
                    break
                else:
                    if seg[0] >= seg[1]:
                        video_seg.append(video[seg[0]])
                    else:
                        video_seg.append(torch.matmul(hand_score[seg[0]:seg[1]].softmax(dim=0), video[seg[0]:seg[1]]))

            if not self.egovlp:
                video_seg = torch.stack(video_seg) # n, 768
            video_seg = torch.matmul(score, video_seg) # 768 (weight combination)
                
            question = self.mlp_t(question)
            
            state = self.state
            scores = []
            for i, actions_per_step in enumerate(actions): # in actions, all the actions in the all steps are stored. such as 3 steps * 14 actions
                a_texts, a_buttons = zip(*[(action['text'], action['button']) for action in actions_per_step]) # action: dic{text: ..., button: ...}
                # a_texts: Num_actions (A), 768
                a_texts = self.mlp_t(torch.cat(a_texts))
                A = len(a_buttons)
                a_buttons = self.mlp_v(
                    torch.stack(a_buttons).view(A, -1, a_texts.shape[1]) 
                ).view(A, -1) # A, 768
                qa = question + a_texts # A, 768

                inputs = torch.cat(
                    [video_seg.expand(A, -1),  text_seg.expand(A, -1), qa.view(A, -1), a_buttons.view(A, -1)],
                    dim=1
                )
                inputs = self.mlp_pre(inputs) # A, 768
                if hasattr(self, "gru"):
                    states = self.gru(inputs, state.expand_as(inputs)) # inputs, hidden_states
                else:
                    states = torch.cat([inputs, state.expand_as(inputs)], dim=1)
                # states A, 768
                logits = self.proj(states) # logits 14, 1
                if self.training:
                    loss += F.cross_entropy(logits.view(1, -1), label[i].view(-1))
                    count += 1
                else:
                    scores.append(logits.view(-1).tolist())
                if self.history_train == "gt" and self.training: # train
                    state = inputs[label[i]]
                if (self.history_train == "max" and self.training) \
                    or (self.history_val == "max" and not self.training):
                    state = inputs[logits.argmax()]
            if not self.training:
                meta["scores"] = scores
                results.append(meta)
        if self.training:
            return loss / count
        else:
            return results

models = {"q2a_function_blip": Q2A_Function_blip, "q2a_interaction_global": Q2A_Interaction_Global,
          "q2a_interaction_clip": Q2A_Interaction_clip, "q2a_interaction_local": Q2A_Interaction_Local}


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = models[cfg.MODEL.ARCH](cfg)
        self.cfg = cfg
        self.validation_step_outputs = []
    
    def training_step(self, batch, idx):
        loss = self.model(batch)
        dataset = self.trainer.datamodule.__class__.__name__
        self.log(f"{dataset} loss", loss, rank_zero_only=True)
        return loss
    
    def configure_optimizers(self):
        cfg = self.cfg
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        return [optimizer], []
    
    def validation_step(self, batch, idx):
        batched_results = self.model(batch)
        self.validation_step_outputs.append(batched_results)
        return batched_results
        
    def on_validation_epoch_end(self, *arg, **kwargs) -> None:
        from eval_for_loveu_cvpr2023 import evaluate
        results = sum(self.validation_step_outputs, [])
        self.validation_step_outputs.clear()  # free memory
        all_preds = {}
        for result in results:
            pred = dict(
                question=result['question'], 
                scores=result['scores']
            )
            folder = result['folder']
            if folder not in all_preds:
                all_preds[folder] = []
            all_preds[folder].append(pred)

        if self.cfg.DATASET.GT:
            with open(self.cfg.DATASET.GT) as f:
                all_annos = json.load(f)
            r1, r3, mr, mrr = evaluate(all_preds, all_annos)
            dataset = self.trainer.datamodule.__class__.__name__
            # for tensorboard
            self.log(f"{dataset} recall@1", r1, rank_zero_only=True)
            self.log(f"{dataset} recall@3", r3, rank_zero_only=True)
            self.log(f"{dataset} mean_rank", mr, rank_zero_only=True)
            self.log(f"{dataset} mrr", mrr)
            # for terminal
            print(f"{dataset} recall@1", r1)
            print(f"{dataset} recall@3", r3)
            print(f"{dataset} mean_rank", mr)
            print(f"{dataset} mrr", mrr) 
        else:
            json_name = f"submit_test_{self.current_epoch}e.json"
            json_file = os.path.join(self.logger.log_dir, json_name)
            if not os.path.exists(self.logger.log_dir):
                os.makedirs(self.logger.log_dir)
            print("\n No ground-truth labels for validation \n")
            print(f"Generating json file at {json_file}. You can zip and submit it to CodaLab ;)")
            with open(json_file, 'w') as f:
                json.dump(all_preds, f)

def build_model(cfg):
    return ModelModule(cfg)
