from pathlib import Path


def make_pathlist_voc(
    dataset_path: Path
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
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
    ) -> tuple[list[str], list[str]]:
        """ファイル名のリストからパスリストを作成
        
        Args:
            data_names: ファイル名のリスト
            
        Returns:
            list: 画像パスリスト
            list: アノテーションパスリスト
        """
        img_dir_path = dataset_path.joinpath("JPEGImages")
        anno_dir_path = dataset_path.joinpath("Annotations")
        
        img_path_list, anno_path_list = [], []
        for name in data_names:
            # 空の文字列はスキップ
            if name == "":
                continue
            
            img_path_list.append(img_dir_path.joinpath(f"{name}.jpg"))
            anno_path_list.append(anno_dir_path.joinpath(f"{name}.xml"))
            
        return img_path_list, anno_path_list
    
    # 訓練用、検証用のデータをそれぞれ取得
    data_names_path = dataset_path.joinpath("ImageSets", "Main")
    
    ## 訓練用
    with open(data_names_path.joinpath("train.txt"), mode="r", encoding="utf-8") as ft:
        train_data_names = ft.read().split("\n")
        train_img_list, train_anno_list = image_and_annotation_path(train_data_names)    
        
    ## 検証用
    with open(data_names_path.joinpath("val.txt"), mode="r", encoding="utf-8") as fv:
        val_data_names = fv.read().split("\n")
        val_img_list, val_anno_list = image_and_annotation_path(val_data_names)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list
    
    
    
