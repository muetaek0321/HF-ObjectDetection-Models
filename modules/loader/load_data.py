from pathlib import Path

import pandas as pd


def make_pathlist_voc(
    dataset_path: Path,
    is_split: bool = False,
    test_data_ratio: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """データセットフォルダを読み込み、画像とアノテーションのパスリストを作成
       (VOCDataset用)
       
    Args:
        dataset_path (str): データセットフォルダのルートパス
    
    Returns:
        list: 訓練用の画像パスリスト
        list: 訓練用のアノテーションパスリスト
        list: 検証用の画像パスリスト
        list: 検証用のアノテーションパスリスト
    """
    
    def image_and_annotation_path(
        data_names: list[str]
    ) -> pd.DataFrame:
        """ファイル名のリストからパスリストを作成
        
        Args:
            data_names: ファイル名のリスト
            
        Returns:
            list: 画像パスリスト
            list: アノテーションパスリスト
        """
        img_dir_path = dataset_path.joinpath("JPEGImages")
        anno_dir_path = dataset_path.joinpath("Annotations")
        
        input_data_dict = {"image": [], "annotation": []}
        for name in data_names:
            # 空の文字列はスキップ
            if name == "":
                continue
            
            input_data_dict["image"].append(img_dir_path.joinpath(f"{name}.jpg"))
            input_data_dict["annotation"].append(anno_dir_path.joinpath(f"{name}.xml"))
            
        return pd.DataFrame(input_data_dict)
    
    # 訓練用、検証用のデータをそれぞれ取得
    data_names_path = dataset_path.joinpath("ImageSets", "Main")
    
    ## 訓練用
    with open(data_names_path.joinpath("train.txt"), mode="r", encoding="utf-8") as ft:
        train_data_names = ft.read().split("\n")
        train_df = image_and_annotation_path(train_data_names)    
        
    ## 検証用
    with open(data_names_path.joinpath("val.txt"), mode="r", encoding="utf-8") as fv:
        val_data_names = fv.read().split("\n")
        val_df = image_and_annotation_path(val_data_names)
        
    # 検証用の一部をテスト用に回す
    if is_split:
        split = int(len(val_df)*test_data_ratio)
        val_df_sp = val_df.iloc[:split, :]
        test_df = val_df.iloc[split:, :]
        
        return train_df, val_df_sp, test_df
    
    else:
        return train_df, val_df
    
    
    
