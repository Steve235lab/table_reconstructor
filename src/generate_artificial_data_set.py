import os
import random

import pandas as pd
from PIL import Image, ImageFont, ImageChops
from handright import Template, handwrite

from cell_text_classifier import get_road_sections_by_road_name


def text_to_image(text: str, output_path: str, image_num: int = 100):
    if output_path[-1] != "/":
        output_path += "/"
    if not os.path.exists(output_path + text):
        os.mkdir(output_path + text)
    for i in range(image_num):
        images = handwrite(text=text, template=template, seed=random.random())
        for _, img in enumerate(images):
            if isinstance(img, Image.Image):
                inverted_img = ImageChops.invert(img.convert("RGB"))
                box = inverted_img.getbbox()
                cropped_im = img.crop(box)
                # im.show()
                cropped_im.save(output_path + text + "/{}.jpg".format(i))


def generate_road_section_data(road_name: str):
    output_path = "/Users/steve235lab/Documents/DataSets/artificial_CN_handwriting/road_section/" + road_name
    os.mkdir(output_path)
    road_sections = get_road_sections_by_road_name(road_name)
    road_sections.apply(func=text_to_image, args=(output_path, 100))


template = Template(
    background=Image.new(mode="1", size=(900, 1000), color=1),
    font=ImageFont.truetype("../font/QUIETSKY.ttf", size=100),
    line_spacing=150,
    fill=0,  # 字体“颜色”
    left_margin=0,
    top_margin=0,
    right_margin=0,
    bottom_margin=0,
    word_spacing=5,
    line_spacing_sigma=6,  # 行间距随机扰动
    font_size_sigma=10,  # 字体大小随机扰动
    word_spacing_sigma=3,  # 字间距随机扰动
    start_chars="“（[<",  # 特定字符提前换行，防止出现在行尾
    end_chars="，。",  # 防止特定字符因排版算法的自动换行而出现在行首
    perturb_x_sigma=4,  # 笔画横向偏移随机扰动
    perturb_y_sigma=4,  # 笔画纵向偏移随机扰动
    perturb_theta_sigma=0.05,  # 笔画旋转偏移随机扰动
)

# 生成"管孔材质"列的数据
# text_list = ["硅", "硅管", "钢", "钢管", "波纹管", "波", "PVC", "PE"]
# output_path = "/Users/steve235lab/Documents/DataSets/artificial_CN_handwriting/col_material/"

# 生成"光、电缆权属（客户信息）"列的数据
# text_list = ["移动", "联通", "信息", "未知", "有线"]
# output_path = "/Users/steve235lab/Documents/DataSets/artificial_CN_handwriting/col_client/"

# 生成表头部分"路名"数据
# road_name_table = pd.read_excel("../references/road_names_of_sh.xlsx", sheet_name="Sheet1")
# text_list = road_name_table["路名"].unique()
# print(len(text_list))
# output_path = "/Users/steve235lab/Documents/DataSets/artificial_CN_handwriting/road_name/"

# if __name__ == "__main__":
#     for text in text_list:
#         if not os.path.exists(output_path + text):
#             os.mkdir(output_path + text)
#         for i in range(100):
#             images = handwrite(text=text, template=template, seed=random.random())
#             for _, img in enumerate(images):
#                 if isinstance(img, Image.Image):
#                     inverted_img = ImageChops.invert(img.convert("RGB"))
#                     box = inverted_img.getbbox()
#                     cropped_im = img.crop(box)
#                     # im.show()
#                     cropped_im.save(output_path + text + "/{}.jpg".format(i))


# 生成表头部分"路段"数据
road_name_table = pd.read_excel("../references/road_names_of_sh.xlsx", sheet_name="Sheet1")
road_name_list = pd.Series(road_name_table["路名"].unique())
road_name_list.apply(generate_road_section_data)
