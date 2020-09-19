
from pycocotools.coco import COCO
import os.path as osp
import numpy as np
import random,mmcv

ann_file='/home/ubuntu/data/global-wheat-detection/folds_v2/0/coco_tile_train.json'


class load_json_wheat:
    CLASSES = ("wheat",)
    def __init__(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        self.data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            self.data_infos.append(info)


    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        return img_info,ann_info

    def __len__(self):
        return len(self.data_infos)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


class Mosaic_mixup_kaggle(object):
    """
    Mosaic or mixup
    """

    def __init__(self, json_path, prob=0.7, mprob=0.5, min_size=4, max_aspect_ratio=10, pad_val=0):
        self.json_path = json_path
        self.prob = prob
        self.mprob = mprob
        self.wheat_jsons = {'train': load_json_wheat(self.json_path[0]),
                            'crops_fold': load_json_wheat(self.json_path[1]),
                            'SPIKE_images': load_json_wheat(self.json_path[2])
                            }
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
        self.pad_val = tuple(pad_val)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        else:
            if random.random() < self.mprob:
                if 'crops_fold' in results["img_prefix"]['roots'][0]:
                    wheat_json = self.wheat_jsons['crops_fold']
                if 'SPIKE_images' in results["img_prefix"]['roots'][0]:
                    wheat_json = self.wheat_jsons['SPIKE_images']
                if 'train' in results["img_prefix"]['roots'][0]:
                    wheat_json = self.wheat_jsons['train']
                h, w = results['img'].shape[:2]

                xc, yc = [int(random.uniform(w * 0.25, w * 0.75)),
                          int(random.uniform(h * 0.25, h * 0.75))]  # center x, y
                indexes = [-1] + [random.randint(0, len(wheat_json) - 1) for _ in range(3)]

                result_image = np.full((h, w, 3), self.pad_val, dtype=np.uint8)
                result_boxes = []

                for i, index in enumerate(indexes):
                    if index != -1:
                        image, boxes = wheat_json.prepare_train_img(index)
                        img_prefix = np.random.choice(results["img_prefix"]["roots"],
                                                      p=results["img_prefix"]["probabilities"])
                        filename = osp.join(img_prefix, image["filename"])
                        file_client = mmcv.FileClient()
                        img_bytes = file_client.get(filename)
                        img = mmcv.imfrombytes(img_bytes, flag='color')
                        boxes = boxes['bboxes']
                    else:
                        img = results['img']
                        boxes = results['gt_bboxes']
                    if i == 0:
                        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h,0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                    elif i == 1:  # top right
                        x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w), yc
                        x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                    elif i == 2:  # bottom left
                        x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h, yc + h)
                        x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
                    elif i == 3:  # bottom right
                        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w), min(h, yc + h)
                        x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                    result_image[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
                    padw = x1a - x1b
                    padh = y1a - y1b

                    boxes[:, 0] += padw
                    boxes[:, 1] += padh
                    boxes[:, 2] += padw
                    boxes[:, 3] += padh

                    result_boxes.append(boxes)

                result_boxes = np.concatenate(result_boxes, 0)
                result_boxes[:, [0, 2]] = result_boxes[:, [0, 2]].clip(0, w)
                result_boxes[:, [1, 3]] = result_boxes[:, [1, 3]].clip(0, h)

                result_boxes = result_boxes[
                    np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
                # 排除极端box
                aspect_ratios = self.calculate_aspect_ratios(result_boxes)
                widths = result_boxes[:, 2] - result_boxes[:, 0]
                heights = result_boxes[:, 3] - result_boxes[:, 1]
                size_mask = (widths > self.min_size) & (heights > self.min_size)
                aspect_ratio_mask = aspect_ratios < self.max_aspect_ratio
                mask = size_mask * aspect_ratio_mask
                result_boxes = result_boxes[mask]

                labels = np.zeros((result_boxes.shape[0],), dtype=np.int64)
                results['img'] = result_image
                results['gt_bboxes'] = result_boxes
                results['gt_labels'] = labels
            else:
                if 'crops_fold' in results["img_prefix"]['roots'][0]:
                    wheat_json = self.wheat_jsons['crops_fold']
                if 'SPIKE_images' in results["img_prefix"]['roots'][0]:
                    wheat_json = self.wheat_jsons['SPIKE_images']
                if 'train' in results["img_prefix"]['roots'][0]:
                    wheat_json = self.wheat_jsons['train']
                index = random.randint(0, len(wheat_json) - 1)
                image, boxes = wheat_json.prepare_train_img(index)
                img_prefix = np.random.choice(results["img_prefix"]["roots"], p=results["img_prefix"]["probabilities"])
                filename = osp.join(img_prefix, image["filename"])
                file_client = mmcv.FileClient()
                img_bytes = file_client.get(filename)
                img = mmcv.imfrombytes(img_bytes, flag='color')
                boxes = boxes['bboxes']

                # get min shape
                h_a, w_a = results["img"].shape[:2]
                h_b, w_b = img.shape[:2]
                max_h = max(h_a, h_b)
                max_w = max(w_a, w_b)
                a = np.full((max_h, max_w, 3), self.pad_val, dtype=np.uint8)
                a[:h_a, :w_a] = results['img']
                b = np.full((max_h, max_w, 3), self.pad_val, dtype=np.uint8)
                b[:h_b, :w_b] = img

                # print(f"resultsize:{results['img'].shape},imgsize:{img.shape}")

                results["img"] = ((a + b) / 2).astype(results["img"].dtype)
                results['gt_bboxes'] = np.concatenate([results['gt_bboxes'], boxes], axis=0)
                results['gt_labels'] = np.zeros((results['gt_bboxes'].shape[0],), dtype=np.int64)
                results["img_shape"] = (max_h, max_w)

            return results

    def calculate_aspect_ratios(self, bboxes, eps=1e-6):
        return np.maximum(
            (bboxes[:, 2] - bboxes[:, 0]) / (bboxes[:, 3] - bboxes[:, 1] + eps),
            (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0] + eps),
        )




# a=load_json()
# ann_file='/home/ubuntu/data/global-wheat-detection/folds_v2/0/coco_tile_train.json'
# a.load_annotations(ann_file)
# pass
data_root='/home/ubuntu/data/global-wheat-detection/'
json_path=  [data_root+ "folds_v2/0/coco_tile_train.json",
            data_root + "folds_v2/0/coco_pseudo_train.json",
            data_root + "coco_spike.json"]

results={'img_prefix':{'roots':['/home/ubuntu/data/global-wheat-detection/SPIKE_images',
                                    "/home/ubuntu/data/global-wheat-detection/stylized_SPIKE_images_v1/",
                                    "/home/ubuntu/data/global-wheat-detection/stylized_SPIKE_images_v2/",
                                    "/home/ubuntu/data/global-wheat-detection/stylized_SPIKE_images_v3/",
                                    "/home/ubuntu/data/global-wheat-detection/stylized_SPIKE_images_v4/"
                                 ],"probabilities":[0.7, 0.3 / 4, 0.3 / 4, 0.3 / 4, 0.3 / 4]},'img':np.ones((1024,1024,3),dtype=np.uint8),
         'anno_info':{'bboxes':np.ones((1,4),dtype=np.float64)},'gt_bboxes':np.ones((1,4),dtype=np.float64)}

a=Mosaic_mixup_kaggle(json_path,prob=0.8,pad_val=(0,0,0))
b=a(results)
pass