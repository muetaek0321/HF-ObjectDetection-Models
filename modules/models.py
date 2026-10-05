import json
from pathlib import Path

import torch
from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DetrConfig,
    DetrForObjectDetection,
    RfDetrConfig,
    RfDetrForObjectDetection,
    RTDetrV2Config,
    RTDetrV2ForObjectDetection,
)
from transformers.modeling_utils import PreTrainedModel


def get_model_train(
    model_name: str, classes: list[str], lr_backbone: float, use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """使用するモデルの準備"""
    if model_name == "DETR":
        return detr(classes, lr_backbone, use_pretrained)
    elif model_name == "Deformable-DETR":
        return deformable_detr(classes, lr_backbone, use_pretrained)
    elif model_name == "RF-DETR":
        return rf_detr(classes, lr_backbone, use_pretrained)
    elif model_name == "RT-DETR":
        return rt_detr(classes, lr_backbone, use_pretrained)


def detr(
    classes: list[str], lr_backbone: float, use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """DETRモデルを準備"""
    id2label = {str(i): class_name for i, class_name in enumerate(classes)}
    label2id = {class_name: i for i, class_name in enumerate(classes)}
    if use_pretrained:
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]
    else:
        config = DetrConfig(id2label=id2label, label2id=label2id)
        model = DetrForObjectDetection(config)
        params = model.parameters()

    return model, params


def deformable_detr(
    classes: list[str], lr_backbone: float, use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """Deformable-DETRモデルを準備"""
    id2label = {str(i): class_name for i, class_name in enumerate(classes)}
    label2id = {class_name: i for i, class_name in enumerate(classes)}
    if use_pretrained:
        model = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]
    else:
        config = DeformableDetrConfig(id2label=id2label, label2id=label2id)
        model = DeformableDetrForObjectDetection(config)
        params = model.parameters()

    return model, params


def rf_detr(
    classes: list[str], lr_backbone: float, use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """RF-DETRモデルを準備"""
    id2label = {str(i): class_name for i, class_name in enumerate(classes)}
    label2id = {class_name: i for i, class_name in enumerate(classes)}
    if use_pretrained:
        model = RfDetrForObjectDetection.from_pretrained(
            "Roboflow/rf-detr-small",
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]
    else:
        config = RfDetrConfig(id2label=id2label, label2id=label2id)
        model = RfDetrForObjectDetection(config)
        params = model.parameters()

    return model, params


def rt_detr(
    classes: list[str], lr_backbone: float, use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """RT-DETRモデルを準備"""
    id2label = {str(i): class_name for i, class_name in enumerate(classes)}
    label2id = {class_name: i for i, class_name in enumerate(classes)}
    if use_pretrained:
        model = RTDetrV2ForObjectDetection.from_pretrained(
            "PekingU/rtdetr_v2_r50vd",
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": lr_backbone,
            },
        ]
    else:
        config = RTDetrV2Config(id2label=id2label, label2id=label2id)
        model = RTDetrV2ForObjectDetection(config)
        params = model.parameters()

    return model, params


def get_model_inference(
    model_name: str, train_result_path: str | Path, device: str | torch.device
) -> PreTrainedModel:
    """推論で使用するモデルの準備"""
    # モデルのコンフィグの読み込み
    with open(train_result_path.joinpath("config.json"), mode="r", encoding="utf-8") as f:
        # モデルのconfigを読み込み
        model_cfg = json.load(f)

    # モデルアーキテクチャの読み込み
    if model_name == "DETR":
        config = DetrConfig(**model_cfg)
        model = DetrForObjectDetection(config)
    elif model_name == "Deformable-DETR":
        config = DeformableDetrConfig(**model_cfg)
        model = DeformableDetrForObjectDetection(config)
    elif model_name == "RF-DETR":
        config = RfDetrConfig(**model_cfg)
        model = RfDetrForObjectDetection(config)
    elif model_name == "RT-DETR":
        config = RTDetrV2Config(**model_cfg)
        model = RTDetrV2ForObjectDetection(config)

    # 学習済みモデルパラメータを読み込み
    weight_path = next(train_result_path.glob("*best.pth"))
    model.load_state_dict(torch.load(weight_path, map_location=device))

    return model
