import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('0_不同的融合方法/yolo11n-midfusion-concat.yaml')

    # 断点续训
    # model = YOLO("runs/train/yolo11n-midfusion/weights/last.pt")

    # 不能cache
    model.train(data='datasets/DroneVehicle/DV.yaml',
                imgsz=640,
                epochs=200,
                batch=16,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                # amp=False, # close amp
                use_simotm="RGBT",
                channels=4,
                # resume=True,  # 断点续训
                project='runs/train',
                name='DV-yolo11n-mid-concat'
                )
