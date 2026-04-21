from pathlib import Path

import cv2
import gdown
import numpy as np
import torch
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.model_utils import fuse_model
from engageConfig import *
import time
from boxmot import BotSort
from boxmot import ByteTrack
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights)


# Dictionary for YOLOX model weights URLs
YOLOX_ZOO = {
    'yolox_n.pt': 'https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX',
    'yolox_s.pt': 'https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj',
    'yolox_m.pt': 'https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun',
    'yolox_l.pt': 'https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz',
    'yolox_x.pt': 'https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5',
}



# Initialize tracker
#tracker = BotSort(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device, half=False)

class Tracking:
    def __init__(self,objectDetector='fastercnn',tracker='ByteTrack'):
        # device = torch.device('cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.objectDetector = objectDetector
        if objectDetector=='yolo':
            self.initializeYoloDetetcor()
        else:
            self.initializeDetector()
        self.initializeTracker()
    def initializeDetector(self):
        # Load detector with pretrained weights and preprocessing transforms
        weights = Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        self.detector.to(self.device).eval()
        self.transform = weights.transforms()
    def initializeYoloDetetcor(self):
        # Preprocessing pipeline
        #self.input_size = [800, 1440]
        self.input_size = [640, 640]

        yolox_model_path = Path(yolox_model)

        # Download model if not present
        if not yolox_model_path.exists():
            gdown.download(YOLOX_ZOO[yolox_model], output=str(yolox_model_path), quiet=False)

        # Initialize YOLOX model
        exp = get_exp(None, 'yolox_x')
        exp.num_classes = 1
        ckpt = torch.load(yolox_model_path, map_location=self.device)

        self.detector = exp.get_model()
        self.detector.load_state_dict(ckpt["model"])
        self.detector = fuse_model(self.detector).to(self.device).eval()
    def initializeTracker(self):
        self.tracker = ByteTrack(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=self.device, half=False)

    # This preprocess differs from the current version of YOLOX preprocess, but ByteTrack uses it
    # https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/data/data_augment.py#L189
    def yolox_preprocess(self,image,
                         input_size,
                         mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225),
                         swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0

        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean

        if std is not None:
            padded_img /= std

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def detect(self,frame):
        if self.objectDetector=='yolo':
            # Preprocess frame
            frame_img, ratio = self.yolox_preprocess(frame, input_size=self.input_size)
            frame_tensor = torch.Tensor(frame_img).unsqueeze(0).to(self.device)

            # Detection with YOLOX
            with torch.no_grad():
                dets = self.detector(frame_tensor)
            dets = postprocess(dets, 1, 0.5, 0.7, class_agnostic=True)[0]

            if dets is not None:
                # Rescale coordinates from letterbox back to the original frame size
                dets[:, 0] = (dets[:, 0]) / ratio
                dets[:, 1] = (dets[:, 1]) / ratio
                dets[:, 2] = (dets[:, 2]) / ratio
                dets[:, 3] = (dets[:, 3]) / ratio
                dets[:, 4] *= dets[:, 5]
                dets = dets[:, [0, 1, 2, 3, 4, 6]].cpu().numpy()
                #===if the bbox center is less than 780, do ...
                dets[:,3] = np.where(((dets[:,1]+dets[:,3])/2)<780,np.minimum(np.full_like(dets[:,3],780),dets[:,3] ),dets[:,3])

            else:
                dets = np.empty((0, 6))
        else:
            # Convert frame to RGB and prepare for detector
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
            input_tensor = self.transform(tensor).to(self.device)

            # Run detection
            output = self.detector([input_tensor])[0]
            scores = output['scores'].detach().cpu().numpy()
            labels = output['labels'].detach().cpu().numpy()
            keep = labels == 1  # scores >= 0.5

            # Prepare detections for tracking
            boxes = output['boxes'][keep].detach().cpu().numpy()
            #===expand y axis===
            boxes[:,3] = boxes[:,3]*1.1
            labels = output['labels'][keep].detach().cpu().numpy()
            filtered_scores = scores[keep]
            dets = np.concatenate([boxes, filtered_scores[:, None], labels[:, None]], axis=1)

        #===sort it from left to right
        #dets = np.asarray(sorted(dets, key=lambda x: x[0]))
        sorted_indices = dets[:, 0].argsort()
        sorted_dets = dets[sorted_indices]
        return sorted_dets

    def track(self,dets,frame,show_lost=False):
        # Update tracker
        res = self.tracker.update(dets, frame)

        # =======trackers_data===#
        if self.tracker.per_class_active_tracks is None:  # dict
            active_tracks = self.tracker.active_tracks
        else:
            active_tracks = []
            for k in self.tracker.per_class_active_tracks.keys():
                active_tracks += self.tracker.per_class_active_tracks[k]

        trackers_data =[]
        for a in active_tracks:
            if not a.history_observations:
                continue
            if len(a.history_observations) < 3:
                continue
            box = a.history_observations[-1]

            state = self.tracker._infer_state(a)

            if state != 'confirmed':
                print(f"state {state}")
            t_id = a.id
            trackers_data.append((a.id,box,state))

        return trackers_data

    def plot_boxes_on_img(
            self,
            img: np.ndarray,
            trackersInfo,
            thickness: int = 2,
            fontscale: float = 0.5,
            state: str = "confirmed",
            style: str = "solid",  # "solid" | "dashed" (dashed only for AABB)
            ) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.
        """

        for id,box in trackersInfo:
            color = self.tracker.id_to_color(id, state=state)
            x1, y1, x2, y2 = map(int, (box[0], box[1], box[2], box[3]))
            if style == "dashed":
                img = self._draw_dashed_rect(img, x1, y1, x2, y2, color, thickness)
            else:
                img = cv2.rectangle(img,(x1, y1),(x2, y2),color,thickness,)
            img = cv2.putText(
                img,f"id: {int(id)}",
                (x1, max(0, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX,fontscale,color,thickness,)
        return img

