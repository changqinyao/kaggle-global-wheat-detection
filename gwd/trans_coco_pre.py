import torch

num_classes = 1
model_coco = torch.load("/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/dumps/DetectoRS_R50-0f1c8080.pth")

# weight
model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
                                                        :num_classes, :]
model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
                                                        :num_classes, :]
model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
                                                        :num_classes, :]
# bias
model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][:num_classes]
# save new model
torch.save(model_coco, "/home/ubuntu/PycharmProjects/kaggle-global-wheat-detection/dumps/DetectoRS_coco_pretrained_weights_classes_%d.pth" % num_classes)