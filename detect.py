import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Caminho para o modelo treinado")
    parser.add_argument("--video", type=str, help="Caminho para o arquivo de vídeo")
    args = parser.parse_args()

    model = YOLO(args.model)
    for result in model.predict(source=args.video, stream=True):
        result.show()

if __name__ == "__main__":
    main()