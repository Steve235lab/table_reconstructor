import pandas as pd


def get_road_names() -> pd.Series:
    return pd.read_excel("../references/road_names_of_sh.xlsx", sheet_name="Sheet1")["路名"].unique()


def get_road_sections_by_road_name(road_name: str) -> pd.Series:
    road_table = pd.read_excel("../references/road_names_of_sh.xlsx", sheet_name="Sheet1")
    road_sections = pd.Series(road_table[road_table["路名"] == road_name]["路段"].unique())
    return road_sections


class CellTextClassifier:
    def __init__(self, cell_type: str, net: str = "ResNet101", road_name: str = None):
        self.net = net
        self.cell_type = cell_type
        self.cell_type_class_num_dict = {
            "col_material": 5,
            "col_client": 5,
            "road_name": 15795,
        }
        self.data_set_path = "D:/DataSets/artificial_CN_handwriting/"
        self.road_name = road_name
        if self.cell_type == "col_material":
            self.class_list = ["硅", "硅管", "钢", "钢管", "波纹管", "波", "PVC", "PE"]
        elif self.cell_type == "col_client":
            self.class_list = ["移动", "联通", "信息", "未知", "有线"]
        elif self.cell_type == "road_name":
            self.class_list = get_road_names()
        elif self.cell_type == "road_section":
            self.class_list = get_road_sections_by_road_name(self.road_name)
