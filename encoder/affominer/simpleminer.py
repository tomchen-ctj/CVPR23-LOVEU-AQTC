from torchvision.ops import box_iou

from .hircnn import HandInteractionRCNN

from pytorch_lightning import LightningDataModule, LightningModule, Trainer


class AffoMiner(LightningModule):
    def __init__(self, min_cont_affo_frames=2,
        max_side_frames=31, max_num_hands=2, 
        hand_state_nms_thresh=0.5, 
        contact_state_threshold=0.99,
        fps=5):
        super().__init__()
        self.hand_state_detector = HandInteractionRCNN(
            box_detections_per_img=max_num_hands)
        self.hand_state_detector.eval()
        self.min_cont_affo_frames = min_cont_affo_frames
        self.max_side_frames = max_side_frames
        self.hand_state_nms_thresh = hand_state_nms_thresh
        self.contact_state_threshold = contact_state_threshold
        self.fps = fps
        
    def hand_state_nms(self, boxes, states, scores):
        # if has P and N overlap, then maintain the highest score one
        if len(boxes) == 2 and states.sum() == 1:
            iou = box_iou(boxes[0,None],boxes[1,None])[0]
            # scores have been ranked
            if iou > self.hand_state_nms_thresh:
                boxes, states, scores = boxes[0,None], states[0,None], scores[0,None]
        return boxes, states, scores
    
    def test_step_per_video(self, video):

        hand_bboxes, hand_states, hand_scores = self.hand_state_detector(video)
        hand_bboxes, hand_states, hand_scores = self.hand_state_nms(hand_bboxes, hand_states, hand_scores)
            
            # judge contacting or not
        frame = dict(hand_bboxes=hand_bboxes, hand_states=hand_states, hand_scores=hand_scores)
        return frame