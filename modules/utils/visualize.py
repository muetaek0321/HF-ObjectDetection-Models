from pathlib import Path

import numpy as np
import supervision as sv

from modules.utils import imwrite_jpn


def visualize_bbox(
    img: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    classes: list[str],
    output_path: str | Path = "image_vis.png",
    save: bool = True,
) -> np.ndarray:
    """BBoxを可視化した画像を出力

    Args:
        img (np.ndarray): 画像データ
        bboxes (np.ndarray,list): BBoxが格納された配列
        labels (np.ndarray,list): ラベルが格納された配列
        scores (np.ndarray,list): スコアが格納された配列
        classes (list[str]): クラス名のリスト
        output_path (str | Path): 出力パス
        save (bool): 画像を保存するかどうか
    """
    # supervisionのDetectionsクラスに変換
    detections = sv.Detections(
        xyxy=bboxes,
        confidence=scores,
        class_id=labels,
    )

    # BBoxの描画
    box_annotator = sv.BoxAnnotator()
    img_vis = box_annotator.annotate(
        scene=img,
        detections=detections,
    )

    # ラベルの描画
    label_annotator = sv.LabelAnnotator()
    img_vis = label_annotator.annotate(
        scene=img_vis,
        detections=detections,
        labels=[
            f"{classes[class_id]} {score:.2f}"
            for class_id, score in zip(labels, scores, strict=True)
        ],
    )

    if save:
        # 可視化画像を保存
        imwrite_jpn(output_path, img_vis)

    return img_vis
