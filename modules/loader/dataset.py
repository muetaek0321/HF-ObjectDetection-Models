"""
参考：
・アノテーション読み込みクラス
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations.core.bbox_utils import normalize_bboxes
from transformers.image_transforms import corners_to_center_format

from .augmentation import get_transform
from modules.utils import imread_jpn


__all__ = ["DETRDataset"]


class AnnoXmlToList(object):
    """VOC形式のデータセットのXMLファイルの読み込みクラス"""

    def __init__(
        self, 
        classes: list[str]
    ) -> None:
        """コンストラクタ
        
        Args:
            classes (list): 検出するクラス名のリスト
        """
        self.classes = classes

    def __call__(
        self, 
        xml_path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """アノテーションファイルを読み込み
        
        Args:
            xml_path (str,Path): アノテーションファイルのパス
            
        Returns:
            tuple: アノテーションデータ(bbox, label)
        """
        bbox_datas = []
        label_datas = []

        # xmlファイル（アノテーションファイル）を読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体（object）の数だけループする
        for obj in xml.iter('object'):

            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            name = obj.find('name').text.lower().strip()  # 物体名
            bbox = obj.find('bndbox')  # バウンディングボックスの情報

            # アノテーションの xmin, ymin, xmax, ymaxを取得
            # VOCは原点が(1,1)なので1を引き算して（0, 0）に
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = [int(bbox.find(pt).text)-1 for pt in pts]
            bbox_datas.append(bndbox)

            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            label_datas.append(label_idx)

        return np.array(bbox_datas), np.array(label_datas)    


class DETRDataset(Dataset):
    """DETR用データセットクラス"""
    
    def __init__(
        self,
        img_path_list: list[Path],
        anno_path_list: list[Path],
        classes: list[str],
        input_size: list[int],
        dataset_type: str,
        phase: str
    ) -> None:
        """コンストラクタ
        
        Args:
            img_path_list (list): 画像パスリスト
            anno_path_list (list): アノテーションパスリスト
            dataset_type (str): データセット形式
            phase (str): 訓練、検証、テストの指定
        """
        self.img_path_list = img_path_list
        self.anno_path_list = anno_path_list
        self.input_size = input_size
        self.dataset_type = dataset_type
        self.phase = phase
        
        # アノテーション変換の準備
        if self.dataset_type == "coco":
            pass
        elif self.dataset_type == "pascal_voc":
            self.load_anno = AnnoXmlToList(classes)
        
        # DataAugmentationの準備
        self.transform = get_transform(self.dataset_type, self.input_size, self.phase)
        
    def __len__(
        self
    ) -> int:
        """Datasetの長さを返す"""
        return len(self.img_path_list)
    
    def __getitem__(
        self, 
        index: int
    ) -> tuple[np.ndarray, dict]:
        """index指定でデータを返す
        
        Args:
            index (int): Datasetのインデックス
            
        Returns:
            np.ndarray: 画像
            list: アノテーション
        """
        # 画像とアノテーションのパスを取得
        img_path, anno_path = self.img_path_list[index], self.anno_path_list[index]
        
        # 画像読み込み
        img = imread_jpn(img_path)
        
        # アノテーションの読み込み
        if self.dataset_type == "coco":
            pass
        elif self.dataset_type == "pascal_voc":
            bboxes, labels = self.load_anno(anno_path)
            
        # DataAugmentationの適用
        transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        img_trans = transformed['image']
        bboxes_trans = transformed['bboxes']
        labels_trans = transformed['labels']
                
        # BBoxを正規化
        h, w = self.input_size
        bboxes_trans = corners_to_center_format(np.array(bboxes_trans))
        boxes_norm = normalize_bboxes(bboxes_trans, rows=h, cols=w)
        
        # モデルの入力形式に変換  
        targets = {
            "image_id": torch.tensor([index]),
            "boxes": torch.as_tensor(boxes_norm, dtype=torch.float32),
            "class_labels": torch.as_tensor(labels_trans, dtype=torch.long),
        }
        
        return {"pixel_values": img_trans, "labels": targets}
        
        
        

