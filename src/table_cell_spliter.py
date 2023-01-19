import os

import cv2
import numpy as np

from ml2xlsx import ML2Xlsx


class Spliter:
    def __init__(self, img_path: str, use_onmt: bool = True):
        """单元格拆分模块，输入原始的表格图片，以 index_column.png 的格式输出若干切分后的单元格图片文件。

        :param img_path: 表格图片路径
        :param use_onmt: 是否使用 onmt Im2Text 进行辅助拆分
        """
        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_binary = cv2.Canny(self.img_gray, 100, 200)    # 上下阈值需要根据实际样本合理设定或者考虑动态调整
        self.height, self.width = self.img_binary.shape
        img_name = img_path.split('/')[-1]
        self.output_path = '../output/cell_split/' + img_name + '/'    # 用图片名作为输出目录
        # 检查输出目录是否存在，若不存在则创建目录
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        # 使用onmt Im2Text 预测表格结构
        if use_onmt:
            # 存放原始图片的目录（不含文件名）
            src_dir = ''
            for cache in img_path.split('/')[:-1]:
                src_dir += cache + '/'
            src_dir = src_dir[:-1]
            # 在 temp 目录下生成一个临时文件并写入图片的文件名，用于 onmt_translate
            src = '../temp/temp_src_' + img_name + '.txt'
            temp_src_file = open(src, 'w')
            temp_src_file.write(img_name)
            temp_src_file.close()
            # 指定 onmt_translate 结果的输出位置
            onmt_output_path = '../temp/onmt_output_' + img_name + '.txt'
            # 拼接 onmt Im2Text 调用指令
            onmt_Im2Text_cmd = "onmt_translate -data_type img " \
                                    "-model E:/SteveCodeBase/Python/table_reconstructor/checkpoints/recognition/Recognition_All.pt " \
                                    "-src_dir " + src_dir + " " \
                                    "-src " + src + " " \
                                    "-output " + onmt_output_path + " " \
                                    "-max_length 150 -beam_size 5 -gpu 0 -verbose -batch_size 8"
            # 使用 onmt Im2Text 对表格结构进行预测
            os.system(onmt_Im2Text_cmd)
            # pred = open(onmt_output_path, 'r')
            # print(pred.readline())
            # pred.close()
            # 将预测得到的表格结构转为 DataFrame 对象
            m2x = ML2Xlsx(onmt_output_path)
            self.table_structure_df = m2x.ml2xlsx()[0]
            if not self.table_structure_df.empty:
                self.rows, self.cols = self.table_structure_df.shape
            else:
                self.rows = None
                self.cols = None
            # 删除临时文件
            os.remove(src)
            os.remove(onmt_output_path)
        else:
            self.rows = None
            self.cols = None

    def get_cross_point(self, scale_x: int = 20, scale_y: int = 20):
        while True:
            # 识别横线
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.width // scale_x, 1))
            eroded = cv2.erode(self.img_binary, kernel, iterations=1)
            dilated_col = cv2.dilate(eroded, kernel, iterations=1)
            cv2.imshow("表格横线展示：", dilated_col)
            cv2.waitKey()
            # 识别竖线
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.height // scale_y))
            eroded = cv2.erode(self.img_binary, kernel, iterations=1)
            dilated_row = cv2.dilate(eroded, kernel, iterations=1)
            cv2.imshow("表格竖线展示：", dilated_row)
            cv2.waitKey()
            # 标识交点
            bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
            cv2.imshow("表格交点展示：", bitwise_and)
            cv2.waitKey()
            # 标识表格
            merge = cv2.add(dilated_col, dilated_row)
            cv2.imshow("表格整体展示：", merge)
            cv2.waitKey()
            # 两张图片进行减法运算，去掉表格框线
            merge2 = cv2.subtract(self.img_binary, merge)
            cv2.imshow("图片去掉表格框线展示：", merge2)
            cv2.waitKey()
            # 识别黑白图中的白色交叉点，将横纵坐标取出
            ys, xs = np.where(bitwise_and > 0)
            ys = np.append(ys, 0)
            ys = np.append(ys, self.height)
            xs = np.append(xs, 0)
            xs = np.append(xs, self.width)
            cross_x = {}  # 横坐标
            cross_y = {}  # 纵坐标
            # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，只取相近值的最后一点
            my_xs = np.sort(xs)
            for i in range(len(my_xs) - 1):
                # 以 坐标: 该坐标与后一坐标差值 的形式保存
                cross_x[my_xs[i]] = my_xs[i + 1] - my_xs[i]
            # cross_x[0] = 1e9    # 加入0
            # cross_x[self.width] = 1e9  # 要将最后一个点加入
            my_ys = np.sort(ys)
            for i in range(len(my_ys) - 1):
                cross_y[my_ys[i]] = my_ys[i + 1] - my_ys[i]
            # cross_y[0] = 1e9
            # cross_y[self.height] = 1e9  # 要将最后一个点加入
            if len(cross_x) >= self.cols + 1 and len(cross_y) >= self.rows + 1:
                return cross_x, cross_y
            if len(cross_x) < self.cols + 1:
                scale_x *= 2
            if len(cross_y) < self.rows + 1:
                scale_y *= 2

    def split(self):
        cross_x, cross_y = self.get_cross_point()
        # print('cross_y', cross_y)
        # print('cross_x', cross_x)
        # 对坐标差值进行排序
        diff_x_list = list(cross_x.values())
        diff_x_list.sort(reverse=True)
        diff_y_list = list(cross_y.values())
        diff_y_list.sort(reverse=True)
        # 取出差值前 self.rows + 1 大的 y 和 前 self.cols + 1 大的 x
        my_x_list = []
        my_y_list = []
        for i in range(self.rows + 1):
            for key, value in cross_y.items():
                if value == diff_y_list[i] and key not in my_y_list:
                    my_y_list.append(key)
                    break
        for i in range(self.cols + 1):
            for key, value in cross_x.items():
                if value == diff_x_list[i] and key not in my_x_list:
                    my_x_list.append(key)
                    break
        my_x_list.sort()
        my_y_list.sort()
        print(my_x_list)
        print(my_y_list)
        for i in range(len(my_y_list)-1):
            for j in range(len(my_x_list)-1):
                cell_img = self.img[my_y_list[i]:my_y_list[i+1], my_x_list[j]:my_x_list[j+1]]
                cv2.imshow("cell", cell_img)
                cv2.waitKey()


if __name__ == "__main__":
    test_spliter = Spliter("D:/DataSets/TableBank/Recognition/images/%c3%89pid%c3%a9miologie%20du%20Diab%c3%a8te+PNL)_1.png")
    # cv2.imshow("二值化图片：", test_spliter.img_binary)
    # cv2.waitKey()
    # print(test_spliter.onmt_Im2Text_cmd)
    print(test_spliter.rows)
    print(test_spliter.cols)
    test_spliter.split()
