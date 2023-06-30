import torch
from torch import nn
from pytorch_lightning import LightningModule
import timm, os
from transformers import AutoModel
from egovlp.egovlp import EgoVLP
import sys
import parse_config
from affominer.simpleminer import AffoMiner


def get_hand_states(hand_state):
    d = []
    for hand in hand_state:
        if hand.numel() == 0:
            d.append(-1)
        elif 1 in hand:
            d.append(1)
        else:
            d.append(0)
    return torch.Tensor(d).cuda()


def get_vision_and_text_model(cfg):
    model_dict = {
                'blip':'Salesforce/blip-itm-large-coco',
                'clip':['vit_large_patch14_clip_336.openai_ft_in12k_in1k', 'xlnet-base-cased'],
                'egovlp': 'vit_large_patch14_clip_336.openai_ft_in12k_in1k' # button & local
                  }
    
    if 'blip' in cfg.MODEL.VISION:
        model = AutoModel.from_pretrained(model_dict['blip'], mirror='tuna')
        return model.vision_model, model.text_model #blip: huggingface
    
    elif 'clip' in cfg.MODEL.VISION:
        model_vision = timm.create_model(model_dict['clip'][0], pretrained=True) # timm
        model_vision.head = nn.Identity()
        model_text = AutoModel.from_pretrained(model_dict['clip'][1], mirror='tuna') # huggingface
        return model_vision, model_text
    
    elif 'egovlp' in cfg.MODEL.VISION:
        model_text = build_egovlp().model.compute_text
        if 'local' in cfg.MODEL.VISION:
            model_vision = build_egovlp().encode_video # egovlp_local: pure egovlp
            return model_vision, model_text # 
        
        elif 'global' in cfg.MODEL.VISION: # egovlp_global: video_global:egovlp, video_local:clip, button:clip, text:egovlp
            if cfg.FOR.GLOBAL and cfg.FOR.VIDEO:
                model_vision = build_egovlp().encode_video
                return model_vision, model_text
            
            else:
                model_vision = timm.create_model(model_dict['egovlp'], pretrained=True)
                model_vision.head = nn.Identity()                
                return model_vision, model_text
    else:
        return None
    
        


class Encoder(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.vision_model, self.text_model = get_vision_and_text_model(cfg)
        
        
        self.for_video = cfg.FOR.VIDEO
        self.for_script = cfg.FOR.SCRIPT
        self.for_qa = cfg.FOR.QA
        self.for_para = cfg.FOR.PARA
        self.for_hand = cfg.FOR.HAND
        self.for_global = cfg.FOR.GLOBAL # global
        self.for_local = cfg.FOR.LOCAL # local
        
        self.model = cfg.MODEL.VISION
        
        if self.for_hand:
            self.detect = AffoMiner().eval()
        print(self.vision_model)


    def test_step(self, batch, idx):
        if batch[0] is None:
            return 

        if self.for_hand:
            video, path = batch[0]
            hands_state = []
            for frame in video:
                hands_state.append(self.detect.test_step_per_video(frame)['hand_states'])
            state = get_hand_states(hands_state)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(state, os.path.join(path, 'hands.pth'))
        
        if self.for_video:
            
            if self.for_local: # local case
                video, path, timestamps = batch[0]
                features = torch.cat([
                    self.vision_model(frame.expand(4, -1, -1, -1).unsqueeze(0))
                    for frame in video
                ])

                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(features, os.path.join(path, 'video_egovlp_local.pth'))

            elif self.for_global: # global case
                video, path, timestamps = batch[0]
                # video T, 3, 224, 224
                # for each timestamp: list [start, end]
                # case a.  current clip's length is less than 16
                # case b.  current clip's length is larger than 16 : divide (//16) -> pooling 
                
                video_clips = [] # all of the clip segments saved in this list
                for seg in timestamps:
                    clip_len = seg[1] - seg[0]
                    if clip_len == 0:
                        video_clip = self.vision_model(video[seg[1]:seg[1]+2].unsqueeze(0)).squeeze()
                    elif 0< clip_len  <= 15:
                        video_clip = self.vision_model(video[seg[0]:seg[1]].unsqueeze(0)).squeeze()
                    elif clip_len > 16:
                        # print(timestamps, seg)
                        video_clip = []
                        clips = clip_len // 16
                        for i in range(clips):
                            video_clip.append(self.vision_model(video[seg[0]+i*16:seg[0]+i*16+15].unsqueeze(0)).squeeze())
                        if seg[1] - (seg[0]+i*16+15)  >= 1:
                            video_clip.append(self.vision_model(video[seg[0]+i*16+15: seg[1]].unsqueeze(0)).squeeze())
                        video_clip = torch.stack(video_clip)
                        if video_clip.shape[0] > 1:
                            video_clip = video_clip.mean(dim=0) # pooling
                    video_clips.append(video_clip.squeeze())
                features = torch.stack(video_clips)

                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(features, os.path.join(path, 'video_egovlp_global.pth'))    


            else:
                video, path, timestamps = batch[0] 
                # set smaller batch size to prevent OOM   

                # for timm models clip & egovlp_global's local part
                if self.model == 'clip' or 'egovlp_global':
                    features = torch.cat([
                        self.vision_model(frame.unsqueeze(0))
                        for frame in video
                    ])

                # for transformers blip:
                elif self.model == 'blip':
                    features = torch.cat([
                        self.vision_model(frame.unsqueeze(0)).last_hidden_state[:, 0, :]
                        for frame in video
                    ])

                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(features, os.path.join(path, "video.pth"))

        if self.for_script:
            if 'egovlp' in self.model:
                script, timestamps, path = batch[0]
                if not os.path.exists(path):
                    os.makedirs(path)
                features = torch.cat([self.text_model(sentence) for sentence in script])
                torch.save([timestamps, features], os.path.join(path, "script.pth"))   
            else:             
                script, timestamps, path = batch[0]
                if not os.path.exists(path):
                    os.makedirs(path)
                features = torch.cat([self.text_model(sentence.input_ids, attention_mask = sentence.attention_mask).last_hidden_state[:,0,:] for sentence in script])
                torch.save([timestamps, features], os.path.join(path, "script.pth"))
              
            
        if self.for_para:
            if 'egovlp' in self.model:
                timestamps, paras, path = batch[0]
                if not os.path.exists(path):
                    os.makedirs(path)
                features = torch.cat([self.text_model(sentence) for sentence in paras])
                torch.save([timestamps, features], os.path.join(path, "paras.pth"))   
            else:             
                timestamps, paras, path = batch[0]
                if not os.path.exists(path):
                    os.makedirs(path)
                features = torch.cat([self.text_model(sentence.input_ids, attention_mask = sentence.attention_mask).last_hidden_state[:,0,:] for sentence in paras])
                torch.save([timestamps, features], os.path.join(path, "paras.pth"))


        if self.for_qa:
            qas, path, tag = batch[0]
            if not os.path.exists(path):
                os.makedirs(path)
            for qa in qas:
                if 'egovlp' in self.model:
                    qa['question'] = self.text_model(qa['question'])
                else:
                    qa['question'] = self.text_model(qa['question'].input_ids, attention_mask=qa['question'].attention_mask).last_hidden_state[:,0,:]
                button_features = []
                for button_images_per_step in qa['button_images']: 
                    
                    if 'blip' in self.model: # huggingface transformers vision model (blip)
                        button_features.append(
                            [
                                self.vision_model(button_image.view(-1,3,button_image.shape[-2], button_image.shape[-1])).last_hidden_state[:, 0, :].flatten() \
                                for button_image in button_images_per_step
                            ]
                        )
                    
                    elif 'clip' in self.model or self.for_global: # timm vision model (clip, egovlp_global)
                        button_features.append(
                            [
                                self.vision_model(button_image.view(-1,3,button_image.shape[-2], button_image.shape[-1])).flatten() \
                                for button_image in button_images_per_step
                            ]
                        )
                        
                    else:  # egovlp_local
                        button_features.append(
                            [
                                self.vision_model(button_image.view(-1,3,button_image.shape[-2], button_image.shape[-1]).expand(4, -1, -1, -1).unsqueeze(0)).flatten() \
                                for button_image in button_images_per_step
                            ]
                        )
                for i, answers_per_step in enumerate(qa['answers']):
                    for j, answer in enumerate(answers_per_step):
                        bidx = qa['answer_bidxs'][i][j]
                        button_feature = button_features[i][bidx]
                        if 'egovlp' in self.model:
                            text_feature = self.text_model(answer)    
                        else:                 
                            text_feature = self.text_model(answer.input_ids, attention_mask = answer.attention_mask).last_hidden_state[:,0,:]
                        answer_feature = dict(text=text_feature, button=button_feature)
                        qa['answers'][i][j] = answer_feature
            torch.save(qas, os.path.join(path, f'{tag}.pth'))


def build_model(cfg):
    return Encoder(cfg)

def build_egovlp():
    """
    return the visual part of the egovlp
    """
    from egovlp.egovlp import EgoVLP
    model = EgoVLP()
    model.eval()
    return model

if __name__ == '__main__':
    pass