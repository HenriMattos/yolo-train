from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="dataset/dataset.yaml", epochs=30, imgsz=640)
