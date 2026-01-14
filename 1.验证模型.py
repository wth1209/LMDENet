import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO("runs/train/DV消融实验/base-HBN-DCEM/weights/best.pt")
    model.val(data='datasets/DroneVehicle/DV.yaml',
              split='val',
              imgsz=640,
              project='runs/val',
              name='DV-HBN-DCEM',
              use_simotm="RGBT",  # RGB、RGBT
              channels=4,  # 3、4
              save=True,  # 是否保存结果
              # save_txt=True,  # 保存标签,用于绘制TP、FP、FN
              )
