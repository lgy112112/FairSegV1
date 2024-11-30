import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd

join = os.path.join
import torch
from torch.utils.data import Dataset
import random
import glob


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpzTrainSet(Dataset):
    def __init__(self, data_root, bbox_shift=10, image_size=1024, random=True):
        self.data_root = data_root
        self.npz_files = sorted(glob.glob(join(data_root, "*.npz")))
        self.bbox_shift = bbox_shift
        self.image_size = image_size
        self.random = random
        print(f"number of images: {len(self.npz_files)}")

    def __len__(self):
        return len(self.npz_files)

    def convert_label(self, label_data, chosen_id):
        # convert label id from (-2, -1, 0) to (0, 1, 2)
        # -2/2: optic cup, -1/1: optic rim, 0: background
        label_data = np.abs(label_data)
        # segment optic cup only
        if chosen_id == 2:
            label_data = np.where(label_data == 2, 1, 0)
        # segment the whole disc (cup and rim)
        if chosen_id == 1:
            label_data = np.where(label_data > 0, 1, 0)
        if chosen_id == 'all':
            lb0 = np.where(label_data == 0, 1, 0)  # cup
            lb1 = np.where(label_data > 0, 1, 0)  # disc
            label_data = np.stack([lb0, lb1], axis=0)
        return label_data

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.npz_files[index])
        npz_data = np.load(join(self.data_root, img_name), "r", allow_pickle=True)  # (x, y)

        # get image data and label data
        image_data = npz_data["slo_fundus"]
        label_data = npz_data["disc_cup_mask"]

        # load privacy-related metadata
        race = npz_data['race']
        ethnicity = npz_data['ethnicity']
        gender = npz_data['gender']
        language = npz_data['language']

        # stack the image data to create a 3-channel image
        image_data = np.stack([image_data] * 3, axis=-1)

        # resize the image and label to (1024, 1024) using OpenCV API
        # note that the interpolation for label should be nearest
        image_data = cv2.resize(image_data, (self.image_size, self.image_size))
        label_data = cv2.resize(label_data, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # convert the shape to (3, H, W)
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data / 255.0
        assert (
            np.max(image_data) <= 1.0 and np.min(image_data) >= 0.0
        ), "image should be normalized to [0, 1]"
        if self.random:
            chosen_label = random.choice([1, 2])
            gt2D = self.convert_label(label_data, chosen_label)
        else:
            gt2D = self.convert_label(label_data, 'all')

        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        if self.random:
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": torch.tensor(image_data).float(),
            "label": torch.tensor(gt2D[None, :, :]).long() if self.random else torch.tensor(gt2D).long(),
            "bboxes": torch.tensor(bboxes).float() if self.random else torch.tensor(0),
            "img_name": img_name,
            'race': race,
            'gender': gender,
            'language': language,
            'ethnicity': ethnicity
        }


class NpzTestSet(Dataset):
    def __init__(self, data_root, bbox_shift=10, image_size=1024):
        self.data_root = data_root
        self.npz_files = sorted(glob.glob(join(data_root, "*.npz")))
        self.bbox_shift = bbox_shift
        self.image_size = image_size
        print(f"number of images: {len(self.npz_files)}")

    def __len__(self):
        return len(self.npz_files)

    def convert_label(self, label_data, chosen_id):
        # convert label id from (-2, -1, 0) to (0, 1, 2)
        # -2/2: optic cup, -1/1: optic rim, 0: background
        label_data = np.abs(label_data)
        # segment optic cup only
        if chosen_id == 2:
            label_data = np.where(label_data == 2, 1, 0)
        # segment the whole disc (cup and rim)
        if chosen_id == 1:
            label_data = np.where(label_data > 0, 1, 0)
        return label_data

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.npz_files[index])
        npz_data = np.load(join(self.data_root, img_name), "r", allow_pickle=True)  # (x, y)

        # get image data and label data
        image_data = npz_data["slo_fundus"]
        label_data = npz_data["disc_cup_mask"]

        # load privacy-related metadata
        race = npz_data['race']
        ethnicity = npz_data['ethnicity']
        gender = npz_data['gender']
        language = npz_data['language']

        # stack the image data to create a 3-channel image
        image_data = np.stack([image_data] * 3, axis=-1)

        # resize the image and label to (1024, 1024) using OpenCV API
        # note that the interpolation for label should be nearest
        image_data = cv2.resize(image_data, (self.image_size, self.image_size))
        label_data = cv2.resize(label_data, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # convert the shape to (3, H, W)
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data / 255.0
        assert (
            np.max(image_data) <= 1.0 and np.min(image_data) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt_cup = self.convert_label(label_data, 2)
        gt_disc = self.convert_label(label_data, 1)

        assert np.max(gt_cup) == 1 and np.min(gt_cup) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt_cup > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt_cup.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        cup_boxes = np.array([x_min, y_min, x_max, y_max])

        assert np.max(gt_disc) == 1 and np.min(gt_disc) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt_disc > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt_disc.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        disc_boxes = np.array([x_min, y_min, x_max, y_max])

        return {
            "image": torch.tensor(image_data).float(),
            "cup_label": torch.tensor(gt_cup[None, :, :]).long(),
            "disc_label": torch.tensor(gt_disc[None, :, :]).long(),
            "cup_bboxes": torch.tensor(cup_boxes).float(),
            "disc_bboxes": torch.tensor(disc_boxes).float(),
            "img_name": img_name,
            'race': race,
            'gender': gender,
            'language': language,
            'ethnicity': ethnicity
        }



class FairEvaluator:
    def __init__(self, result_df, attribute_df, class_list):
        self.result_df = result_df
        self.att_df = attribute_df
        self.class_list = class_list
    
    def _compute(self, att):
        # given an attribute name, compute the fairness metrics
        item_list = []
        for idx, row in self.att_df.iterrows():
            name = row['name']
            att_info = row[att]
            item = self.result_df.loc[self.result_df['name'] == name]
            metric_info = {k: item[k].values[0] for k in self.class_list}
            item_list.append({'name': name, att: att_info, **metric_info})
        result_df = pd.DataFrame(item_list)

        cls_metric_dict = {}
        for cls in self.class_list:
            cls_metric_dict[cls] = {}
            cls_df = result_df[cls]
            cls_metric_dict[cls]['overall'] = cls_df.mean()
            # get unique vals of att
            att_vals = result_df[att].unique().tolist()
            try:
                att_vals.remove(-1)
            except:
                pass
            for val in att_vals:
                val_df = result_df[result_df[att] == val]
                cls_metric_dict[cls][val] = val_df[cls].mean()
        
        for cls in self.class_list:
            cls_metric = cls_metric_dict[cls]
            diff_list = []
            for k in cls_metric:
                diff_list.append(abs(cls_metric[k] - cls_metric['overall']))
            sum_diff = sum(diff_list)
            es_metric = cls_metric['overall'] / (1 + sum_diff)
            cls_metric_dict[cls][f'es_{cls}'] = es_metric
        return cls_metric_dict

    def __call__(self, attribute):
        return self._compute(attribute)

    def report(self, attribute):
        cls_metric_dict = self._compute(attribute)
        for cls in self.class_list:
            print(f"Class: {cls}")
            print(f"Overall: {cls_metric_dict[cls]['overall']:.4f}")
            print(f"ES: {cls_metric_dict[cls][f'es_{cls}']:.4f}")
            for k in cls_metric_dict[cls]:
                if k != 'overall' and k!= f'es_{cls}':
                    print(f"{k}: {cls_metric_dict[cls][k]:.4f}")
            print("\n")
