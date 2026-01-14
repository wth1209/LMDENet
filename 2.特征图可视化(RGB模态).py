import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')


# 训练好的单模态模型的权重文件
model = YOLO("runs/train/DV对比试验/DV-RGB-yolo11n/weights/best.pt")

# source：数据地址(可以是文件夹、视频)
result = model.predict(source="datasets/DroneVehicle(RGB-T3-T)/train/visible/images/00001.jpg",
                       imgsz=640,
                       show=False,
                       save_frames=True,
                       save=True,  # 保存结果
                       visualize=True,  # 可视化特征图
                       use_simotm="RGB",
                       channels=3,
                       project="runs/predict",
                       name="DV-RGB-yolo11n-train-visible-00001",  # 模型名称加图片编号
                       # conf=0.2,  # 置信度
                       # iou=0.7,  # 交并比
                       # agnostic_nms=True,  # 预测的时候同一个目标出现两个框的解决办法
                       # line_width=2,  # bounding boxes的边界框宽度
                       # show_conf=False,  # 不显示预测的置信度
                       # show_labels=False,  # 不显示预测的标签
                       # save_txt=True,  # 将结果保存为txt文件
                       # save_crop=True,  # 保存裁剪的图像
                       )
