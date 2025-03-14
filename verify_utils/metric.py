import numpy as np
from mmdet3d.datasets import NuScenesDataset
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import  load_gt, add_center_dist,\
    filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import center_distance

from .data_utils import output_to_nusc_box, lidar_nusc_box_to_global


class RobustnessMetric:
    """
    This a base class for computing the query result according to
    the predicted boxes and ground truth boxes on the NuScenes dataset.
    """
    def __init__(self, dataset):

        self.eval_detection_configs = dataset.eval_detection_configs
        self.cls_range = self.eval_detection_configs.class_range
        self.cls_name = dataset.CLASSES

        # according to mmdet3d.datasets.NuScenesDataset
        self.eval_version = dataset.eval_version 
        self.nusc = NuScenes(
            version=dataset.version, dataroot=dataset.data_root,
            verbose=False)

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.data_infos = dataset.data_infos

        self.eval_set = eval_set_map[dataset.version]
        self.gt_boxes = self.get_gt_boxes(self.eval_set)

    def process_nusc_boxes(self, boxes):
        boxes = add_center_dist(self.nusc, boxes)
        boxes = filter_eval_boxes(self.nusc, boxes, self.cls_range, verbose=False)
        return boxes

    def get_gt_boxes(self, eval_set, verbose=False):
        gt_boxes = load_gt(self.nusc, eval_set, DetectionBox, verbose=verbose)
        gt_boxes = self.process_nusc_boxes(gt_boxes)
        return gt_boxes

    def form_pred_boxes(self, result, sample_id, token, class_names):
        results_ = result[0]['pts_bbox']
        boxes = output_to_nusc_box(results_)
        boxes = lidar_nusc_box_to_global(
                                    self.data_infos[sample_id], boxes,
                                    class_names,
                                    self.eval_detection_configs,
                                    self.eval_version)
        pred_boxes = get_pred_boxes_single_frame(boxes, 
                                                 token, 
                                                 class_names)
        pred_boxes = self.process_nusc_boxes(pred_boxes)
        return pred_boxes

    def box_match(self,
                  result: dict,
                  sample_id,
                  dist_th):
        token = self.data_infos[sample_id]['token']
        pred_boxes = self.form_pred_boxes(result, sample_id, 
                                          token, self.cls_name)

        taken = set()
        matched_gt = set()
        for class_name in self.cls_name:
            pred_boxes_list = [box for box in pred_boxes[token] \
                               if box.detection_name == class_name]
            for gt_idx, gt_box in enumerate(self.gt_boxes[token]):
                if gt_box.detection_name != class_name: continue
                min_dist = np.inf
                for pred_idx, pred_box in enumerate(pred_boxes_list):
                    if pred_idx in taken: continue
                    # Find closest match among ground truth boxes
                    this_distance = center_distance(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                is_match = min_dist < dist_th
                if is_match:
                    taken.add(pred_idx)
                    matched_gt.add(gt_idx)
        return matched_gt
    
    def query(self):
        raise NotImplementedError

class CentreDistance(RobustnessMetric):
    """
     Computing the l2 centre distance between the ground truth boxes and
     the predicted boxes
    """
    def query(self,
            result: dict,
            sample_id: int,
            dist_th: float,
            ori_match: set = None,
            negative: bool = True,
            overshoot=1.2):

        token = self.data_infos[sample_id]['token']
        dist_th = dist_th * overshoot
        pred_boxes = self.form_pred_boxes(result, sample_id, 
                                    token, self.cls_name)

        if ori_match is None: ori_match = set()

        pred_boxes_by_class = {}
        for class_name in self.cls_name:
            pred_boxes_by_class[class_name] = \
                [box for box in pred_boxes[token] 
                 if box.detection_name == class_name]

        taken = set()
        matched_gt = set()
        total_dist = 0
        for class_name in self.cls_name:
            pred_boxes_list = pred_boxes_by_class.get(class_name, [])
            pred_centers = get_centers(pred_boxes_list)
            if len(pred_centers) == 0: continue

            taken_mask = np.zeros(len(pred_centers), dtype=bool)
            for gt_idx, gt_box in enumerate(self.gt_boxes[token]):
                if gt_box.detection_name != class_name: continue
                distances = batch_center_distance(gt_box.translation[:2], 
                                                  pred_centers)
                distances[taken_mask] = np.inf

                if distances.min()< dist_th: 
                    total_dist += distances.min()
                    pred_box_ind = np.argmin(distances)
                    taken.add(pred_box_ind)
                    taken_mask[pred_box_ind] = True
                    matched_gt.add(gt_idx)
        # handle mismatching caused by perturbation
        mismatched_count = len(self.gt_boxes[token]) - len(matched_gt)
        total_dist += mismatched_count * dist_th if \
            mismatched_count else 0

        if negative: total_dist =  -1 * total_dist
        return total_dist

class CentreDistance_det(RobustnessMetric):
    """
     Computing the l2 centre distance between the ground truth boxes and
     the predicted boxes
    """
    def query(self,
            result: dict,
            sample_id: int,
            dist_th: float,
            ori_match: set = None,
            negative: bool = True,
            overshoot=1.2):

        token = self.data_infos[sample_id]['token']
        dist_th = dist_th * overshoot
        pred_boxes = self.form_pred_boxes(result, sample_id, 
                                    token, self.cls_name)

        if ori_match is None: ori_match = set()

        pred_boxes_by_class = {}
        for class_name in self.cls_name:
            pred_boxes_by_class[class_name] = \
                [box for box in pred_boxes[token] 
                 if box.detection_name == class_name]

        taken = set()
        matched_gt = set()
        total_dist = 0
        det_score = 0
        for class_name in self.cls_name:
            pred_boxes_list = pred_boxes_by_class.get(class_name, [])
            pred_centers = get_centers(pred_boxes_list)
            if len(pred_centers) == 0: continue

            taken_mask = np.zeros(len(pred_centers), dtype=bool)
            for gt_idx, gt_box in enumerate(self.gt_boxes[token]):
                if gt_box.detection_name != class_name: continue
                distances = batch_center_distance(gt_box.translation[:2], 
                                                  pred_centers)
                distances[taken_mask] = np.inf

                if distances.min()< dist_th: 
                    total_dist += distances.min()
                    pred_box_ind = np.argmin(distances)
                    taken.add(pred_box_ind)
                    taken_mask[pred_box_ind] = True
                    matched_gt.add(gt_idx)
                    det_score += pred_boxes_list[pred_box_ind].detection_score
        # handle mismatching cau ed by perturbation
        mismatched_count = len(self.gt_boxes[token]) - len(matched_gt)
        total_dist += mismatched_count * dist_th if \
            mismatched_count else 0
        final_score = total_dist - det_score

        if negative: final_score =  -1 * final_score
        return final_score

# helper functions
def get_centers(boxes):
    return np.array([box.translation[:2] for box in boxes])

def get_det_score(boxes):
    return np.array([box.detection_score for box in boxes]) 

def batch_center_distance(gt_center, pred_centers):
    return np.linalg.norm(pred_centers - gt_center, axis=1)

def get_pred_boxes_single_frame(nusc_box,
                                sample_token,
                                mapped_class_names):
    nusc_annos = {}
    annos = []
    for i, box in enumerate(nusc_box):
        name = mapped_class_names[box.label]
        if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
            if name in [
                    'car',
                    'construction_vehicle',
                    'bus',
                    'truck',
                    'trailer',
            ]:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = NuScenesDataset.DefaultAttribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = NuScenesDataset.DefaultAttribute[name]

        nusc_anno = dict(
            sample_token=sample_token,
            translation=box.center.tolist(),
            size=box.wlh.tolist(),
            rotation=box.orientation.elements.tolist(),
            velocity=box.velocity[:2].tolist(),
            detection_name=name,
            detection_score=box.score,
            attribute_name=attr,
        )
        annos.append(nusc_anno)
        nusc_annos[sample_token] = annos
    return EvalBoxes.deserialize(nusc_annos, DetectionBox)

