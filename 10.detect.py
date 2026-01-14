import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # image
    '''
        source 需要和train/val目录一致，且需要包含visible字段，visible同级目录下存在infrared目录，
        原理是将visible替换为infrared，加载双光谱数据
    '''
    model = YOLO(r"runs/train/LLVIP消融实验/LLVIP-yolo11n-HBN-DCEM/weights/best.pt")
    model.predict(source=r'datasets/LLVIP(detect)/images/visible/val',
                  imgsz=640,
                  project='runs/detect',
                  name='LMDENet-LLVIP-val',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBT",
                  channels=4,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                  )


    # # VIDEO
    # model = YOLO(R"runs/M3FD/M3FD-yolov5-RGBT-midfusion/weights/best.pt") # select your model.pt path
    # model.predict(source=r"G:\wan\data\RGBT\testVD\visible\video.mp4",
    #               imgsz=640,
    #               project='runs/detect',
    #               name='exp',
    #               show=False,
    #               save_frames=True,
    #               use_simotm="RGBT",
    #               channels=4,
    #               save=True,
    #               # conf=0.2,
    #               # visualize=True # visualize model features maps
    #             )
