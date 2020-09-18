import argparse

import mmcv
from mmcv import Config

from gwd.datasets.wheat_detection import WheatDataset1
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/configs/detectors/detectors_r50_ga_mstrain_stage2.py")
    parser.add_argument("--ann-file", default="/home/ubuntu/data/global-wheat-detection/coco_crops_fold0.json")
    parser.add_argument("--output-path", default="/home/ubuntu/data/global-wheat-detection/folds_v2/0/coco_pseudo_train.json")
    parser.add_argument("--predictions-path", default="/home/ubuntu/data/global-wheat-detection/crops_fold0_predictions.pkl")
    parser.add_argument("--fold",default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_path)
    # cfg.data.test.ann_file = cfg.data.test.ann_file.format(fold=args.fold)
    cfg.data.test.ann_file=args.ann_file
    cfg.data.test.test_mode = True
    dataset: WheatDataset1 = build_dataset(cfg.data.test)
    predictions = mmcv.load(args.predictions_path)
    print(len(predictions), len(dataset))
    dataset.pseudo_results(predictions, output_path=args.output_path, pseudo_confidence_threshold=0.8)


if __name__ == "__main__":
    main()
