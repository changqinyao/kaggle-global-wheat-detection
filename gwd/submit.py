import argparse
import os.path as osp

from gwd import test, wbf
from gwd.converters import images2coco


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_prefix", default="/home/ubuntu/data/global-wheat-detection/test")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--ann_file", default="/home/ubuntu/data/global-wheat-detection/test_coco.json")
    parser.add_argument(
        "--configs",
        default=[
            "/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4.py",
            # "/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/configs/detectors/detectors_r50_ga_mstrain_stage0.py"
            # "configs/universe_spike/universe_r50_mstrain_spike_stage1.py",
        ],
        nargs="+",
    )
    parser.add_argument(
        "--checkpoints",
        default=[
            "/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/mosaic/work_dirs/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4/0/epoch_60.pth",
            # "/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/mosaic/work_dirs/detectors_r50_ga_mstrain_stage0/0/epoch_44.pth"
            # "/dumps/universe_r50_mstrain_spike_stage1_epoch_12.pth",
        ],
        nargs="+",
    )
    parser.add_argument("--weights", default=[3, 1], type=float, nargs="+")
    parser.add_argument("--submission_path", default="/home/ubuntu/data/test_submission.csv")
    parser.add_argument("--pseudo_path", default="/home/ubuntu/data/coco_pseudo_test.json")
    parser.add_argument("--iou_thr", default=0.55, type=float)
    parser.add_argument("--score_thr", default=0.05, type=float)
    return parser.parse_args()


def main(
    img_prefix,
    configs,
    checkpoints,
    weights,
    submission_path,
    ann_file,
    pseudo_path=None,
    format_only=True,
    flip=False,
    iou_thr=0.55,
    score_thr=0.45,
    pseudo_score_threshold=0.8,
    pseudo_confidence_threshold=0.65,
):
    images2coco.main(osp.join(img_prefix, "*"), output_path=ann_file)
    dataset = None
    all_predictions = []
    for config, checkpoint in zip(configs, checkpoints):
        print(f"\nconfig: {config}, checkpoint: {checkpoint}")
        predictions, dataset = test.main(
            config=config,
            checkpoint=checkpoint,
            img_prefix=img_prefix,
            ann_file=ann_file,
            format_only=format_only,
            options=dict(output_path=submission_path),
            flip=flip,
            show=True,
            show_dir='/home/ubuntu/data/',
            show_score_thr=0.05
        )
        all_predictions.append(predictions)
    if len(all_predictions) > 1:
        ensemble_predictions = wbf.main(all_predictions, weights=weights, iou_thr=iou_thr, score_thr=score_thr)
    else:
        ensemble_predictions = all_predictions[0]
    dataset.format_results(results=ensemble_predictions, output_path=submission_path)
    if pseudo_path is not None:
        dataset.pseudo_results(
            results=ensemble_predictions,
            output_path=pseudo_path,
            pseudo_score_threshold=pseudo_score_threshold,
            pseudo_confidence_threshold=pseudo_confidence_threshold,
        )


if __name__ == "__main__":
    main(**vars(parse_args()))
