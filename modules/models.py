import json
from pathlib import Path

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers import DetrConfig, DetrForObjectDetection
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection
from transformers import DetaConfig, DetaForObjectDetection
from transformers import ConditionalDetrConfig, ConditionalDetrForObjectDetection


def get_model_train(
    model_name: str,
    classes: list[str],
    lr_backbone: float,
    use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """使用するモデルの準備
    """
    if model_name == "DETR":
        return detr(classes, lr_backbone, use_pretrained)
    elif model_name == "Deformable-DETR":
        return deformable_detr(classes, lr_backbone, use_pretrained)
    elif model_name == "DETA":
        return deta(classes, lr_backbone, use_pretrained)
    elif model_name == "ConditionalDETR":
        return conditional_detr(classes, lr_backbone, use_pretrained)


def detr(
    classes: list[str],
    lr_backbone: float,
    use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """DETRモデルを準備
    """
    id2label = {str(i): class_name for i, class_name in enumerate(classes)}
    label2id = {class_name: i for i, class_name in enumerate(classes)}
    if use_pretrained:
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", 
            ignore_mismatched_sizes=True,
            id2label=id2label, 
            label2id=label2id
        )
        params = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": lr_backbone},
        ]
    else:
        config = DetrConfig(id2label=id2label, label2id=label2id)
        model = DetrForObjectDetection(config)
        params = model.parameters()
        
    return model, params


def deformable_detr(
    classes: list[str],
    lr_backbone: float,
    use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """Deformable-DETRモデルを準備
    """
    id2label = {str(i): class_name for i, class_name in enumerate(classes+["NONE"])}
    label2id = {class_name: i for i, class_name in enumerate(classes+["NONE"])}
    if use_pretrained:
        model = DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr", 
            ignore_mismatched_sizes=True,
            id2label=id2label, 
            label2id=label2id
        )
        params = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": lr_backbone},
        ]
    else:
        config = DeformableDetrConfig(id2label=id2label, label2id=label2id)
        model = DeformableDetrForObjectDetection(config)
        params = model.parameters()
        
    return model, params


def deta(
    classes: list[str],
    lr_backbone: float,
    use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """DETAモデルを準備
    """
    id2label = {str(i): class_name for i, class_name in enumerate(classes+["NONE"])}
    label2id = {class_name: i for i, class_name in enumerate(classes+["NONE"])}
    if use_pretrained:
        model = DetaForObjectDetection.from_pretrained(
            "jozhang97/deta-resnet-50",
            ignore_mismatched_sizes=True,
            id2label=id2label, 
            label2id=label2id
        )
        params = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": lr_backbone},
        ]
    else:
        config = DetaConfig(id2label=id2label, label2id=label2id)
        model = DetaForObjectDetection(config)
        params = model.parameters()
        
    return model, params


def conditional_detr(
    classes: list[str],
    lr_backbone: float,
    use_pretrained: bool
) -> tuple[PreTrainedModel, list]:
    """Conditional DETRモデルを準備
    """
    id2label = {str(i): class_name for i, class_name in enumerate(classes+["NONE"])}
    label2id = {class_name: i for i, class_name in enumerate(classes+["NONE"])}
    if use_pretrained:
        model = ConditionalDetrForObjectDetection.from_pretrained(
            "microsoft/conditional-detr-resnet-50",
            ignore_mismatched_sizes=True,
            id2label=id2label, 
            label2id=label2id
        )
        params = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": lr_backbone},
        ]
    else:
        config = ConditionalDetrConfig(id2label=id2label, label2id=label2id)
        model = ConditionalDetrForObjectDetection(config)
        params = model.parameters()
        
    return model, params
    

def get_model_inference(
    model_name: str,
    train_result_path: str | Path,
    device: str | torch.device
) -> PreTrainedModel:
    """推論で使用するモデルの準備
    """
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
        
    # 学習済みモデルパラメータを読み込み
    weight_path = list(train_result_path.glob("*best.pth"))[0]
    model.load_state_dict(torch.load(weight_path, map_location=device))
        
    return model
        
    
    
    