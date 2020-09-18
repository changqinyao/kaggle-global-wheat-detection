import os.path as osp
import random
from glob import glob

import cv2
import mmcv
import numpy as np
from albumentations import Compose

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Pad

from .albumentations import albu_builder
from .utils import calculate_area

try:
    from bbaug import policies
except ImportError:
    policies = None


@PIPELINES.register_module()
class RandomRotate90(object):
    def __init__(self, rotate_ratio=None):
        self.rotate_ratio = rotate_ratio
        if rotate_ratio is not None:
            assert 0 <= rotate_ratio <= 1

    def bbox_rot90(self, bboxes, img_shape, factor):
        assert bboxes.shape[-1] % 4 == 0
        h, w = img_shape[:2]
        rotated = bboxes.copy()
        if factor == 1:
            rotated[..., 0] = bboxes[..., 1]
            rotated[..., 1] = w - bboxes[..., 2]
            rotated[..., 2] = bboxes[..., 3]
            rotated[..., 3] = w - bboxes[..., 0]
        elif factor == 2:
            rotated[..., 0] = w - bboxes[..., 2]
            rotated[..., 1] = h - bboxes[..., 3]
            rotated[..., 2] = w - bboxes[..., 0]
            rotated[..., 3] = h - bboxes[..., 1]
        elif factor == 3:
            rotated[..., 0] = h - bboxes[..., 3]
            rotated[..., 1] = bboxes[..., 0]
            rotated[..., 2] = h - bboxes[..., 1]
            rotated[..., 3] = bboxes[..., 2]
        return rotated

    def __call__(self, results):
        if "rotate" not in results:
            rotate = True if np.random.rand() < self.rotate_ratio else False
            results["rotate"] = rotate
        if "rotate_factor" not in results:
            rotate_factor = random.randint(0, 3)
            results["rotate_factor"] = rotate_factor
        if results["rotate"]:
            # rotate image
            for key in results.get("img_fields", ["img"]):
                results[key] = np.ascontiguousarray(np.rot90(results[key], results["rotate_factor"]))
            results["img_shape"] = results["img"].shape
            # rotate bboxes
            for key in results.get("bbox_fields", []):
                results[key] = self.bbox_rot90(results[key], results["img_shape"], results["rotate_factor"])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(factor={self.factor})"


@PIPELINES.register_module()
class RandomCropVisibility(object):
    def __init__(self, crop_size, min_visibility=0.5):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.min_visibility = min_visibility

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results["img_shape"] = img_shape

        if "gt_bboxes" in results:
            original_areas = calculate_area(results["gt_bboxes"])

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get("bbox_fields", []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        if "gt_bboxes" in results:
            gt_bboxes = results["gt_bboxes"]
            gt_bboxes_ignore = results["gt_bboxes_ignore"]
            cropped_areas = calculate_area(gt_bboxes)
            valid_inds = (cropped_areas / original_areas) >= self.min_visibility
            ignore_valid_inds = (gt_bboxes_ignore[:, 2] > gt_bboxes_ignore[:, 0]) & (
                gt_bboxes_ignore[:, 3] > gt_bboxes_ignore[:, 1]
            )
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results["gt_bboxes"] = gt_bboxes[valid_inds, :]
            results["gt_bboxes_ignore"] = gt_bboxes_ignore[ignore_valid_inds, :]
            if "gt_labels" in results:
                results["gt_labels"] = results["gt_labels"][valid_inds]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


class BufferTransform(object):
    def __init__(self, min_buffer_size, p=0.5):
        self.p = p
        self.min_buffer_size = min_buffer_size
        self.buffer = []

    def apply(self, results):
        raise NotImplementedError

    def __call__(self, results):
        if len(self.buffer) < self.min_buffer_size:
            self.buffer.append(results.copy())
            return None
        if np.random.rand() <= self.p and len(self.buffer) >= self.min_buffer_size:
            random.shuffle(self.buffer)
            return self.apply(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nmin_buffer_size={self.min_buffer_size}),\n"
        repr_str += f"(\nratio={self.p})"
        return repr_str


@PIPELINES.register_module()
class Mosaic(BufferTransform):
    """
    Based on https://github.com/dereyly/mmdet_sota
    """

    def __init__(self, min_buffer_size=4, p=0.5, pad_val=0):
        assert min_buffer_size >= 4, "Buffer size for mosaic should be at least 4!"
        super(Mosaic, self).__init__(min_buffer_size=min_buffer_size, p=p)
        self.pad_val = pad_val

    def apply(self, results):
        # take four images
        a = self.buffer.pop()
        b = self.buffer.pop()
        c = self.buffer.pop()
        d = self.buffer.pop()
        # get max shape
        max_h = max(a["img"].shape[0], b["img"].shape[0], c["img"].shape[0], d["img"].shape[0])
        max_w = max(a["img"].shape[1], b["img"].shape[1], c["img"].shape[1], d["img"].shape[1])

        # cropping pipe
        padder = Pad(size=(max_h, max_w), pad_val=self.pad_val)

        # crop
        a, b, c, d = padder(a), padder(b), padder(c), padder(d)

        # check if cropping returns None => see above in the definition of RandomCrop
        if not a or not b or not c or not d:
            return results

        # offset bboxes in stacked image
        def offset_bbox(res_dict, x_offset, y_offset, keys=("gt_bboxes", "gt_bboxes_ignore")):
            for k in keys:
                if k in res_dict and res_dict[k].size > 0:
                    res_dict[k][:, 0::2] += x_offset
                    res_dict[k][:, 1::2] += y_offset
            return res_dict

        b = offset_bbox(b, max_w, 0)
        c = offset_bbox(c, 0, max_h)
        d = offset_bbox(d, max_w, max_h)

        # collect all the data into result
        top = np.concatenate([a["img"], b["img"]], axis=1)
        bottom = np.concatenate([c["img"], d["img"]], axis=1)
        results["img"] = np.concatenate([top, bottom], axis=0)
        results["img_shape"] = (max_h * 2, max_w * 2)

        for key in ["gt_labels", "gt_bboxes", "gt_labels_ignore", "gt_bboxes_ignore"]:
            if key in results:
                results[key] = np.concatenate([a[key], b[key], c[key], d[key]], axis=0)
        return results

    def __repr__(self):
        repr_str = self.__repr__()
        repr_str += f"(\npad_val={self.pad_val})"
        return repr_str


from pycocotools.coco import COCO

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


@PIPELINES.register_module()
class Mosaic_mixup_kaggle(object):
    """
    Mosaic or mixup
    """

    def __init__(self, json_path, prob=0.67, mprob=0.5, min_size=4, max_aspect_ratio=10, pad_val=0):
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

                xc, yc = [int(random.uniform(w * 0.25, w * 0.75)),int(random.uniform(h * 0.25, h * 0.75))]  # center x, y
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
                        h0, w0=img.shape[:2]
                        boxes = boxes['bboxes']
                    else:
                        img = results['img']
                        h0, w0 = img.shape[:2]
                        boxes = results['gt_bboxes']
                    if i == 0:
                        x1a, y1a, x2a, y2a = max(xc - w0, 0), max(yc - h0,0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                        x1b, y1b, x2b, y2b = w0 - (x2a - x1a), h0 - (y2a - y1a), w0, h0  # xmin, ymin, xmax, ymax (small image)
                    elif i == 1:  # top right
                        x1a, y1a, x2a, y2a = xc, max(yc - h0, 0), min(xc + w0, w), yc
                        x1b, y1b, x2b, y2b = 0, h0 - (y2a - y1a), min(w0, x2a - x1a), h0
                    elif i == 2:  # bottom left
                        x1a, y1a, x2a, y2a = max(xc - w0, 0), yc, xc, min(h, yc + h0)
                        x1b, y1b, x2b, y2b = w0 - (x2a - x1a), 0, w0, min(y2a - y1a, h0)
                    elif i == 3:  # bottom right
                        x1a, y1a, x2a, y2a = xc, yc, min(xc + w0, w), min(h, yc + h0)
                        x1b, y1b, x2b, y2b = 0, 0, min(w0, x2a - x1a), min(y2a - y1a, h0)

                    # print(f"resultsize:{result_image.shape},imgsize:{img.shape}")
                    # print(f"xc:{xc},yc{yc},x1a, y1a, x2a, y2a:{x1a, y1a, x2a, y2a},x1b, y1b, x2b, y2b:{x1b, y1b, x2b, y2b}")
                    # print(f"result形状:{result_image[y1a:y2a, x1a:x2a].shape},img形状:{img[y1b:y2b, x1b:x2b].shape}")
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





@PIPELINES.register_module()
class Mixup(BufferTransform):
    def __init__(self, min_buffer_size=2, p=0.5, pad_val=0):
        assert min_buffer_size >= 2, "Buffer size for mosaic should be at least 2!"
        super(Mixup, self).__init__(min_buffer_size=min_buffer_size, p=p)
        self.pad_val = pad_val

    def apply(self, results):
        # take four images
        a = self.buffer.pop()
        b = self.buffer.pop()

        # get min shape
        max_h = max(a["img"].shape[0], b["img"].shape[0])
        max_w = max(a["img"].shape[1], b["img"].shape[1])

        # cropping pipe
        padder = Pad(size=(max_h, max_w), pad_val=self.pad_val)

        # crop
        a, b = padder(a), padder(b)

        # check if cropping returns None => see above in the definition of RandomCrop
        if not a or not b:
            return results

        # collect all the data into result
        results["img"] = ((a["img"].astype(np.float32) + b["img"].astype(np.float32)) / 2).astype(a["img"].dtype)
        results["img_shape"] = (max_h, max_w)

        for key in ["gt_labels", "gt_bboxes", "gt_labels_ignore", "gt_bboxes_ignore"]:
            if key in results:
                results[key] = np.concatenate([a[key], b[key]], axis=0)
        return results


@PIPELINES.register_module()
class RandomCopyPasteFromFile(object):
    def __init__(self, root, n_crops=3, transforms=(), crop_label=1, iou_threshold=0.5, p=0.5):
        self.root = root
        self.n_crops = n_crops
        self.crop_label = crop_label
        self.crop_paths = glob(osp.join(root, "*.jpg"))
        self.aug = Compose([albu_builder(t) for t in transforms])
        self.p = p
        self.iou_threshold = iou_threshold

    def __call__(self, results):
        if np.random.rand() <= self.p:
            for i in range(np.random.randint(1, self.n_crops + 1)):
                filepath = random.choice(self.crop_paths)
                crop = mmcv.imread(filepath)
                crop = self.aug(image=crop)["image"]
                crop_h, crop_w = crop.shape[:2]
                h, w = results["img"].shape[:2]
                if 3 <= crop_w < w and 3 <= crop_h < h:
                    for _ in range(10):
                        y_center = np.random.randint(crop_h // 2, h - crop_h // 2)
                        x_center = np.random.randint(crop_w // 2, w - crop_w // 2)
                        crop_bbox = np.array(
                            [
                                x_center - crop_w // 2,
                                y_center - crop_h // 2,
                                x_center + crop_w // 2,
                                y_center + crop_h // 2,
                            ]
                        ).reshape(1, 4)
                        ious = bbox_overlaps(results["gt_bboxes"], crop_bbox, mode="iof")
                        if max(ious) < self.iou_threshold:
                            crop_mask = 255 * np.ones(crop.shape, crop.dtype)
                            results["img"] = cv2.seamlessClone(
                                src=crop,
                                dst=results["img"],
                                mask=crop_mask,
                                p=(x_center, y_center),
                                flags=cv2.NORMAL_CLONE,
                            )
                            results["gt_bboxes"] = np.concatenate([results["gt_bboxes"], crop_bbox])
                            results["gt_labels"] = np.concatenate(
                                [results["gt_labels"], np.full(len(crop_bbox), self.crop_label)]
                            )
                            break
        return results
