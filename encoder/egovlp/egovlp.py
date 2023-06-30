from __future__ import annotations
import os
import sys
import glob
import collections
import tqdm

import pickle

os.environ['LOCAL_RANK'] = os.getenv('LOCAL_RANK') or '0'

import cv2
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms._transforms_video import NormalizeVideo
import transformers
import sys
import parse_config


device = "cuda" if torch.cuda.is_available() else "cpu"
localfile = lambda *fs: os.path.abspath(os.path.join(os.path.dirname(__file__), *fs))

MODEL_DIR = os.getenv('MODEL_DIR') or 'models' # localfile('../models')

EGOVLP_CHECKPOINT = '/home/hy/ssd1/tomchen/loveu2023/encoder/egovlp/egovlp.pth'



# https://github.com/showlab/EgoVLP/blob/main/configs/eval/epic.json
class EgoVLP(nn.Module):
    norm_mean=(0.485, 0.456, 0.406)
    norm_std=(0.229, 0.224, 0.225)
    def __init__(self, checkpoint=EGOVLP_CHECKPOINT, input_res=224, center_crop=256, n_samples=16, device=device):  #  tokenizer_model="distilbert-base-uncased"
        super().__init__()
        self.q = collections.deque(maxlen=n_samples)
        print(checkpoint)
        
        from model_egovlp.model import FrozenInTime
        model = FrozenInTime(**{
            "video_params": {
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": 16,
                "pretrained": True,
                "time_init": "zeros"
            },
            "text_params": {
                "model": "distilbert-base-uncased",
                "pretrained": True,
                "input": "text"
            },
            "projection": "minimal",
            "load_checkpoint": checkpoint or None,
        })
        ## load model
        #checkpoint = torch.load(checkpoint)
        #state_dict = checkpoint['state_dict']
        ##state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        #model.load_state_dict(state_dict, strict=True)
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
        # image transforms
        self.transforms = T.Compose([
            T.Resize(center_crop),
            T.CenterCrop(center_crop),
            T.Resize(input_res),
            NormalizeVideo(mean=self.norm_mean, std=self.norm_std),
        ])

    def forward(self, video, text, return_sim=True):
        with torch.no_grad():
            text_embed, vid_embed = self.model({'video': video, 'text': text}, return_embeds=True)
            if return_sim:
                return self.similarity(text_embed, vid_embed)
            return vid_embed, text_embed

    def encode_text(self, text, prompt=None):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        with torch.no_grad():
            if self.tokenizer is not None:
                if prompt:
                    text = [prompt.format(t) for t in text]
                text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            return self.model.compute_text({key: val.to(self.device) for key, val in text.items()})

    def encode_video(self, video):
        with torch.no_grad():
            return self.model.compute_video(video)

    def prepare_image(self, im):
        im = im[:,:,::-1]
        im = im.transpose(2, 0, 1)
        im = torch.as_tensor(im.astype(np.float32))[:,None] / 255
        im = self.transforms(im)
        im = im.transpose(1, 0)
        return im

    def add_image(self, im):
        self.q.append(self.prepare_image(im))
        return self

    def predict_recent(self):
        return self.encode_video(torch.stack(list(self.q), dim=1).to(self.device))

    def similarity(self, z_text, z_video, **kw):
        return similarity(z_text, z_video, **kw)

    def few_shot_predictor(self, vocab, vocab_dir='fewshot'):
        vocab_dir = os.path.join(vocab_dir, vocab_dir)
        if os.path.isdir(vocab_dir):
            return FewShotPredictor(vocab_dir)

    def zero_shot_predictor(self, vocab):
        return ZeroShotPredictor(vocab, self)

    def get_predictor(self, vocab, texts=None, vocab_dir='fewshot'):
        pred = None
        if isinstance(vocab, str):
            pred = self.few_shot_predictor(vocab, vocab_dir)
        if pred is None:
            if texts is None and vocab is not None:
                texts = vocab
            if callable(texts):
                texts = texts()
            pred = self.zero_shot_predictor(vocab)
        return pred


def similarity(z_text, z_video, temp=1, temp_inv=1/500, dual=False):
    if dual:
        sim = sim_matrix_mm(z_text, z_video)
        sim = F.softmax(sim*temp_inv, dim=1) * sim
        sim = F.softmax(temp*sim, dim=0)
        return sim.t()
    sim = (sim_matrix(z_text, z_video) + 1) / 2
    return F.softmax(temp*sim.t(), dim=1)


def sim_matrix(a, b, eps=1e-8):
    #added eps for numerical stability
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    return sim_matrix_mm(a_norm, b_norm)

def sim_matrix_mm(a, b):
    return torch.mm(a, b.transpose(0, 1))



class ZeroShotPredictor(nn.Module):
    def __init__(self, vocab, model, prompt='{}'):
        super().__init__()
        self.model = model
        self.vocab = np.asarray(vocab)
        tqdm.tqdm.write(f'zeroshot with: {vocab}')
        self.Z_vocab = self.model.encode_text(vocab, prompt)

    def forward(self, Z_image):
        '''Returns the action probabilities for that frame.'''
        scores = self.model.similarity(self.Z_vocab, Z_image).detach()
        return scores

class FewShotPredictor(nn.Module):
    def __init__(self,  vocab_dir, **kw):
        super().__init__()
        self._load_model(vocab_dir, **kw)

    def _load_model(self, vocab_dir, clsf_type='knn', n_neighbors=33):
        pkl_fname = f'{vocab_dir}_{clsf_type}.pkl'
        if os.path.isfile(pkl_fname):
            with open(pkl_fname, 'rb') as fh:
                tqdm.tqdm.write('loading classifier...')
                self.clsf, self.vocab = pickle.load(fh)
                tqdm.tqdm.write(f'loaded classifier from disk. {len(self.vocab)} {self.vocab}')
            return

        # load all the data
        assert os.path.isdir(vocab_dir)
        fsx = sorted(glob.glob(os.path.join(vocab_dir, 'X_*.npy')))
        fsy = sorted(glob.glob(os.path.join(vocab_dir, 'Y_*.npy')))
        assert all(
            os.path.basename(fx).split('_', 1)[1] == os.path.basename(fy).split('_', 1)[1]
            for fx, fy in zip(fsx, fsy)
        )
        fvocab = os.path.join(vocab_dir, 'classes.npy')

        # load and convert to one big numpy array
        X = np.concatenate([np.load(f) for f in fsx])
        Y = np.concatenate([np.load(f) for f in fsy])
        self.vocab = vocab = np.asarray(np.load(fvocab))
        tqdm.tqdm.write(f'loaded {X.shape} {Y.shape}. {len(vocab)} {vocab}')

        # train the classifier
        tqdm.tqdm.write('training classifier...')
        if clsf_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.clsf = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif clsf_type == 'xgb':
            from xgboost import XGBClassifier
            self.clsf = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
        else:
            raise ValueError(f"Invalid classifier {clsf_type}")
        self.clsf.fit(X, Y)
        tqdm.tqdm.write('trained!')

        with open(pkl_fname, 'wb') as fh:
            pickle.dump([self.clsf, self.vocab], fh)

    def forward(self, Z_image):
        scores = self.clsf.predict_proba(Z_image.cpu().numpy())
        return scores

def l2norm(Z, eps=1e-8):
    return Z / np.maximum(np.linalg.norm(Z, keepdims=True), eps)

def normalize(Z, eps=1e-8):
    Zn = Z.norm(dim=1)[:, None]
    return Z / torch.max(Zn, eps*torch.ones_like(Zn))


def get_predictor(model, vocab, few_shot_dir, step_map=None):
    if os.path.isdir(few_shot_dir):
        tqdm.tqdm.write('few shot')
        predictor = FewShotPredictor(few_shot_dir)
    else:
        tqdm.tqdm.write('zero shot')
        if not vocab:
            raise ValueError(f'no vocab for recipe {os.path.basename(few_shot_dir)}')
        predictor = ZeroShotPredictor(vocab, model)
    if step_map:
        predictor.vocab = np.array([step_map.get(x,x) for x in predictor.vocab])
    return predictor


class ShotEgoVLP(EgoVLP):
    predictor = None
    # def __init__(self, *a, **kw):
    #     super().__init__(*a, **kw)
    
    def set_vocab(self, vocab, few_shot_dir, step_map=None):
        self.predictor = self._get_predictor(vocab, few_shot_dir, step_map)
        assert self.predictor is not None
        self.vocab = self.predictor.vocab

    def _get_predictor(self, vocab, few_shot_dir, step_map=None):
        return get_predictor(self, vocab, few_shot_dir, step_map)
        # if os.path.isdir(few_shot_dir):
        #     tqdm.tqdm.write('few shot')
        #     self.predictor = FewShotPredictor(few_shot_dir)

        # tqdm.tqdm.write('zero shot')
        # if not vocab:
        #     raise ValueError(f'no vocab for recipe {os.path.basename(few_shot_dir)}')
        # self.predictor = ZeroShotPredictor(vocab, self)
        # if step_map:
        #     self.predictor.vocab = np.array([step_map.get(x,x) for x in self.predictor.vocab])

    def predict_video(self, vid):
        assert self.predictor is not None
        z_image = self.model.encode_video(vid)
        sim = self.predictor(z_image)
        return sim




class FrameInput:
    def __init__(self, src, src_fps, fps, give_time=True, fallback_previous=True):
        self.src = src
        self.fps = fps or src_fps
        self.src_fps = src_fps
        self.give_time = give_time
        self.fallback_previous = fallback_previous

    def fname2i(self, f):
        return int(os.path.splitext(os.path.basename(f))[0].split('_')[-1])

    @staticmethod
    def cvt_fps(src_fps, fps):
        return int(max(round(src_fps / (fps or src_fps)), 1))

    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __iter__(self):
        import cv2
        fs = os.listdir(os.path.dirname(self.src))
        i_max = self.fname2i(max(fs))
        every = self.cvt_fps(self.src_fps, self.fps)
        print(f'{self.src}: fps {self.src_fps} to {self.fps}. taking every {every} frames')

        im = None
        for i in tqdm.tqdm(range(0, i_max+1, every)):
            t = i / self.src_fps if self.give_time else i

            f = self.src.format(i)
            if not os.path.isfile(f):
                tqdm.tqdm.write(f'missing frame: {f}')
                if self.fallback_previous and im is not None:
                    yield t, im
                continue

            im = cv2.imread(f)
            yield t, im



if __name__ == '__main__':
    model = EgoVLP()