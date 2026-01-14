import os
import numpy as np
import cv2

# 图片文件夹
img_folder = r"C:\yolo11-mutimodel\datasets\DroneVehicle(RGBT-T)\val\infrared\images"
img_list = os.listdir(img_folder)
img_list.sort()

# 标签文件夹
label_folder = r"C:\yolo11-mutimodel\datasets\DroneVehicle(RGBT-T)\val\infrared\labels"
label_list = os.listdir(label_folder)
label_list.sort()

# 输出图片文件夹位置
output_folder = r"C:\Users\wth\Desktop\123"

# 标签（汽车 货车 卡车 公交车 面包车）
# labels = ['car', 'freight car', 'truck', 'bus', 'van']

# 色盘，为不同类别指定颜色(红、绿、蓝、黄、橙)
colormap = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]


# 坐标转换
def xywh2xyxy(x, w1, h1, img):
    label, x, y, w, h = x

    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1

    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    # # 绘制矩形框
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[1], 2)

    # 给不同目标绘制不同的颜色框
    if int(label) == 0:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[0], 2)
    elif int(label) == 1:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[1], 2)
    elif int(label) == 2:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[2], 2)
    elif int(label) == 3:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[3], 2)
    elif int(label) == 4:
       cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[4], 2)

    return img


if __name__ == '__main__':
    for i in range(len(img_list)):
        image_path = img_folder + "/" + img_list[i]
        label_path = label_folder + "/" + label_list[i]

        # 读取图像文件
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        # 读取 labels
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)

        # 绘制每一个目标
        for x in lb:
            # 反归一化并得到左上和右下坐标，画出矩形框
            img = xywh2xyxy(x, w, h, img)

        """
        # 直接查看生成结果图
        cv2.imshow('show', img)
        cv2.waitKey(0)
        """

        cv2.imwrite(output_folder + '/' + '{}.png'.format(image_path.split('/')[-1][:-4]), img)
