import os.path

import numpy as np
import torch
from ViFiCLIP.utils.config import get_config
from ViFiCLIP.utils.logger import create_logger
from ViFiCLIP.trainers import vificlip
from ViFiCLIP.datasets.pipeline import *
from engageConfig import *
import pandas as pd
# Step 1:
# Configuration class
class parse_option():
    def __init__(self):
        # pretrained_model_path  = './pretrained_action_recog_models/ckpt_epoch_30.pth'
        # config = './pretrained_action_recog_models/16_32_vifi_clip_16_shot_cvpr_n_actions.yaml'
        self.config = config
        self.output =  'exp'   # Name of output folder to store logs and save weights
        self.resume = pretrained_model_path
        # No need to change below args.
        self.only_test = True
        self.opts = None
        self.batch_size = None
        self.pretrained = None
        self.accumulation_steps = None
        self.local_rank = 0

class ActionClassifier:
    def __init__(self, action_names):

        args = parse_option()
        self.config = get_config(args)
        #==create temp folder for logfiles
        if not os.path.exists(args.output):
            os.makedirs(args.output,exist_ok=True)
        # logger
        self.logger = create_logger(output_dir=args.output, name=f"{self.config.MODEL.ARCH}")
        self.logger.info(f"working dir: {self.config.OUTPUT}")

        # # classes_all = pd.read_csv(labels_file, usecols=['id', 'name'])
        # self.class_names = [class_des for i, class_des in classes_all.values.tolist()]
        self.class_names  = action_names

        # Step 2:
        # Create the ViFi-CLIP models and load pretrained weights
        self.classifier = vificlip.returnCLIP(self.config,
                                    logger=self.logger,
                                    class_names=self.class_names, )
        self.classifier = self.classifier.float().cuda()  # changing to cuda here
        self.load_model_checkpoint()
        self.preprocess()
        self.classifier.eval()
    def load_model_checkpoint(self):
        print(f"==============> Resuming form {self.config.MODEL.RESUME}....................")
        checkpoint = torch.load(self.config.MODEL.RESUME, map_location='cpu', weights_only=False)
        load_state_dict = checkpoint['model']
        # now remove the unwanted keys:
        for key in ["module.prompt_learner.token_prefix",  # "prompt_learner.token_prefix",
                    "module.prompt_learner.token_suffix",  # "prompt_learner.token_suffix",
                    "module.prompt_learner.complete_text_embeddings"]:  # ,"prompt_learner.complete_text_embeddings"]:
            if key in load_state_dict:
                del load_state_dict[key]
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # load params
        msg = self.classifier.load_state_dict(new_state_dict, strict=False)
        self.logger.info(f"resume model: {msg}")

    def preprocess(self):
        # Step 3:
        # Preprocessing for video
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        scale_resize = int(256 / 224 * self.config.DATA.INPUT_SIZE)
        val_pipeline = [
            # dict(type='Resize', scale=(-1, scale_resize)),
            # dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
            dict(type='Resize', scale=(self.config.DATA.INPUT_SIZE, self.config.DATA.INPUT_SIZE), keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]
        self.pipeline = Compose(val_pipeline)
    def uniformSampling(self,num_frames):
        # ====uniform sampling====
        inds = self.get_test_clips(num_frames, 16)
        inds = np.mod(inds, num_frames)
        inds = inds.astype(int)
        return inds
    def classify(self,sampledFrms):
        with torch.no_grad():
                with torch.cuda.amp.autocast():
                    dict_file = {'imgs':sampledFrms,'original_shape':sampledFrms[0].shape[:2],'img_shape':sampledFrms[0].shape[:2], 'modality': 'RGB'}
                    video = self.pipeline(dict_file)
                    video_tensor = video['imgs'].unsqueeze(0).cuda().float()
                    logits = self.classifier(video_tensor)
                logits =torch.softmax(logits, dim=1)
                pred_index = logits.argmax(-1)

        return self.class_names[pred_index]
    def get_test_clips(self,num_frames, clip_len, num_clips=1, seed=50):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """

        np.random.seed(seed)
        if num_frames < clip_len:
            # Then we use a simple strategy
            if num_frames < num_clips:
                start_inds = list(range(num_clips))
            else:
                start_inds = [
                    i * num_frames // num_clips
                    for i in range(num_clips)
                ]
            inds = np.concatenate(
                [np.arange(i, i + clip_len) for i in start_inds])
        elif clip_len <= num_frames < clip_len * 2:
            all_inds = []
            for i in range(num_clips):
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
                all_inds.append(inds)
            inds = np.concatenate(all_inds)
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            all_inds = []
            for i in range(num_clips):
                offset = np.random.randint(bsize)
                all_inds.append(bst + offset)
            inds = np.concatenate(all_inds)
        return inds
