import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR


if __name__ == '__main__':
    model = YOLO('0_单模态模型/yolov5n.yaml')
    # model = RTDETR('0_单模态模型/rtdetr-l.yaml')

    # 断点续训
    # model = YOLO("/root/runs/train/LLUAV-yolo11n/weights/last.pt")

    # 不能cache
    model.train(data=r'LLVIP.yaml',
                imgsz=640,
                epochs=200,
                batch=16,
                workers=0,
                device='0',
                optimizer='SGD',
                # amp=False,
                use_simotm="RGB",  # RGB, Gray
                channels=3,  # 3，1
                # resume=True,  # 断点续训
                project='runs/train',
                name='LLVIP-yolov5n'
                )
