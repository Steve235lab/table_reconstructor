import cv2
import imutils


class ImageCorrection:
    def __init__(self, img_path: str, output_path: str, output_height: int = 2432, output_width: int = 1824):
        """图像校正模块
        对原始图像进行边缘裁切、放缩至指定大小

        :param img_path: 原始图像路径（含文件名）
        :param output_path: 输出路径（不含文件名）
        :param output_height: 输出图像高度（单位：像素）
        :param output_width: 输出图像宽度（单位：像素）
        """
        self.img_name = img_path.split('/')[-1]
        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_binary = cv2.Canny(self.img_gray, 100, 200)  # 上下阈值需要根据实际样本合理设定或者考虑动态调整
        self.output_path = output_path
        self.output_height = output_height
        self.output_width = output_width
        self.table_corner = []

    def get_main_body(self):
        # 轮廓检测
        contours = cv2.findContours(self.img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv3() else contours[0]
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # 排序操作，按矩形面积
        max_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
        screenCnt = approx

        # 遍历轮廓
        # for c in contours:
        #     # 计算轮廓近似
        #     peri = cv2.arcLength(c, True)
        #     # C表示输入的点集
        #     # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数，默认百分之二就可以了
        #     # True表示封闭的
        #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 近似成为一个矩形
        #
        #     # 4个点的时候就拿出来
        #     if len(approx) == 4:
        #         screenCnt = approx
        #         break

        # 展示结果
        cv2.drawContours(self.img, [screenCnt], -1, (0, 255, 0), 20)

    def save_img(self, img_name: str, img):
        cv2.imwrite(self.output_path + img_name, img)


if __name__ == "__main__":
    test_img_path = "../test/table_a_0.jpg"
    ic = ImageCorrection(test_img_path, '../output/image_correction/')
    # ic.get_main_body()
    # ic.save_img()
    cv2.imshow('bin', ic.img_binary)
    cv2.waitKey()
