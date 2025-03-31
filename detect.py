import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import threading
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from datetime import datetime
import shutil
import glob
import logging

# Configuração de logging para tratamento de erros
logging.basicConfig(filename="yolo_detection.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class VideoDetectionApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("YOLOv8 Detection Suite Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg="#e0e0e0")
        self.model = self.load_model(model_path)
        self.video_path = None
        self.cap = None
        self.running = False
        self.paused = False
        self.detection_thread = None
        self.run_dir = os.path.dirname(os.path.dirname(model_path))
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.brightness = 0
        self.contrast = 1.0

        self.setup_gui()
        self.load_training_metrics()

    def load_model(self, model_path):
        try:
            model = YOLO(model_path)
            logging.info(f"Modelo carregado com sucesso: {model_path}")
            return model
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao carregar o modelo: {str(e)}")
            self.root.quit()
            return None

    def setup_gui(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Carregar Vídeo", command=self.upload_video)
        file_menu.add_command(label="Salvar Resultados", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.on_closing)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualizar", menu=view_menu)
        view_menu.add_command(label="Imagens do Dataset", command=self.show_dataset_images)

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Frame de vídeo
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Visualização do Vídeo", padding=10)
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.video_frame, width=1280, height=720, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Controles de vídeo
        self.video_controls = ttk.Frame(self.video_frame)
        self.video_controls.pack(pady=5)
        self.pause_button = ttk.Button(self.video_controls, text="Pausar", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.video_controls, text="Brilho:").pack(side=tk.LEFT, padx=5)
        self.brightness_scale = ttk.Scale(self.video_controls, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.update_brightness)
        self.brightness_scale.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.video_controls, text="Contraste:").pack(side=tk.LEFT, padx=5)
        self.contrast_scale = ttk.Scale(self.video_controls, from_=0.5, to=2.0, orient=tk.HORIZONTAL, command=self.update_contrast)
        self.contrast_scale.set(1.0)
        self.contrast_scale.pack(side=tk.LEFT, padx=5)

        # Frame inferior
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

        self.control_frame = ttk.LabelFrame(self.bottom_frame, text="Controle", padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10, "bold"), padding=5)
        self.upload_button = ttk.Button(self.control_frame, text="Carregar Vídeo", command=self.upload_video)
        self.upload_button.pack(fill=tk.X, pady=5)
        self.start_button = ttk.Button(self.control_frame, text="Iniciar Detecção", command=self.start_detection, state=tk.DISABLED)
        self.start_button.pack(fill=tk.X, pady=5)
        self.stop_button = ttk.Button(self.control_frame, text="Parar Detecção", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)

        ttk.Label(self.control_frame, text="Tamanho da Imagem:").pack(pady=(10, 0))
        self.imgsz_var = tk.StringVar(value="640")
        self.imgsz_combo = ttk.Combobox(self.control_frame, textvariable=self.imgsz_var, values=["320", "640", "960", "1280"])
        self.imgsz_combo.pack(fill=tk.X)

        self.conf_var = tk.DoubleVar(value=0.5)
        ttk.Label(self.control_frame, text="Confiança:").pack(pady=(10, 0))
        self.conf_scale = ttk.Scale(self.control_frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL)
        self.conf_scale.pack(fill=tk.X)
        self.conf_label = ttk.Label(self.control_frame, text=f"Confiança: {self.conf_var.get():.2f}")
        self.conf_scale.bind("<Motion>", lambda e: self.conf_label.config(text=f"Confiança: {self.conf_var.get():.2f}"))
        self.conf_label.pack()

        self.save_var = tk.BooleanVar()
        ttk.Checkbutton(self.control_frame, text="Salvar Vídeo", variable=self.save_var).pack(pady=5)

        self.metrics_frame = ttk.LabelFrame(self.bottom_frame, text="Métricas e Depuração", padding=10)
        self.metrics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(self.metrics_frame, height=12, width=40, state=tk.DISABLED, font=("Courier", 10), bg="#ffffff", relief=tk.SUNKEN)
        self.metrics_text.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor="#e0e0e0")
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.metrics_frame)
        self.canvas_graph.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.ax.set_facecolor("#ffffff")
        self.ax.grid(True, linestyle="--", alpha=0.7)

        self.status_var = tk.StringVar(value="Pronto")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_training_metrics(self):
        results_csv = os.path.join(self.run_dir, "results.csv")
        if os.path.exists(results_csv):
            try:
                df = pd.read_csv(results_csv)
                metrics = df.iloc[-1]
                self.update_metrics_text(f"Épocas Totais: {len(df)}\n"
                                        f"mAP@50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}\n"
                                        f"mAP@50:95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}\n"
                                        f"Precisão: {metrics.get('metrics/precision(B)', 'N/A'):.4f}\n"
                                        f"Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}\n"
                                        f"Perda Box: {metrics.get('train/box_loss', 'N/A'):.4f}\n"
                                        f"Perda Classe: {metrics.get('train/cls_loss', 'N/A'):.4f}\n"
                                        f"FPS: {self.fps:.2f}")

                self.ax.clear()
                self.ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label="mAP@50:95", color="#1f77b4")
                self.ax.plot(df['epoch'], df['metrics/precision(B)'], label="Precisão", color="#ff7f0e")
                self.ax.plot(df['epoch'], df['metrics/recall(B)'], label="Recall", color="#2ca02c")
                self.ax.set_xlabel("Época", fontsize=10)
                self.ax.set_ylabel("Métrica", fontsize=10)
                self.ax.legend(fontsize=8)
                self.ax.grid(True, linestyle="--", alpha=0.7)
                self.ax.set_facecolor("#ffffff")
                self.canvas_graph.draw()
                logging.info("Métricas de treinamento carregadas com sucesso.")
            except Exception as e:
                self.update_metrics_text(f"Erro ao carregar métricas: {str(e)}")
                logging.error(f"Erro ao carregar métricas: {str(e)}")
        else:
            self.update_metrics_text("Arquivo results.csv não encontrado.")
            logging.warning("Arquivo results.csv não encontrado.")

    def update_metrics_text(self, text):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.config(state=tk.DISABLED)

    def upload_video(self):
        try:
            self.video_path = filedialog.askopenfilename(filetypes=[("Arquivos de Vídeo", "*.mp4 *.avi *.mov")])
            if self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
                if self.cap.isOpened():
                    self.status_var.set(f"Vídeo carregado: {os.path.basename(self.video_path)}")
                    self.start_button.config(state=tk.NORMAL)
                    self.pause_button.config(state=tk.DISABLED)
                    messagebox.showinfo("Sucesso", "Vídeo carregado com sucesso!")
                    logging.info(f"Vídeo carregado: {self.video_path}")
                else:
                    raise ValueError("Não foi possível abrir o vídeo.")
            else:
                logging.info("Nenhum vídeo selecionado.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar vídeo: {str(e)}")
            self.cap = None
            self.status_var.set("Erro ao carregar vídeo")
            logging.error(f"Erro ao carregar vídeo: {str(e)}")

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Continuar" if self.paused else "Pausar")
        self.status_var.set("Pausado" if self.paused else "Detecção em andamento...")

    def update_brightness(self, value):
        self.brightness = float(value)

    def update_contrast(self, value):
        self.contrast = float(value)

    def adjust_frame(self, frame):
        """Aplica ajustes de brilho e contraste"""
        adjusted = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        return adjusted

    def start_detection(self):
        if not self.model:
            messagebox.showerror("Erro", "Modelo não carregado.")
            logging.error("Tentativa de iniciar detecção sem modelo carregado.")
            return
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Erro", "Nenhum vídeo carregado.")
            logging.error("Tentativa de iniciar detecção sem vídeo carregado.")
            return

        self.running = True
        self.paused = False
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.NORMAL)
        self.status_var.set("Detecção em andamento...")
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_thread = threading.Thread(target=self.detect_video, daemon=True)
        self.detection_thread.start()
        logging.info("Detecção iniciada.")

    def detect_video(self):
        output_path = None
        out = None
        if self.save_var.get():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"output_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(self.cap.get(3)), int(self.cap.get(4))))
                logging.info(f"Vídeo de saída configurado: {output_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao configurar salvamento: {str(e)}")
                logging.error(f"Erro ao configurar salvamento: {str(e)}")
                self.running = False

        while self.running and self.cap.isOpened():
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                start_frame_time = time.time()
                frame = self.adjust_frame(frame)
                imgsz = int(self.imgsz_var.get())
                conf = self.conf_var.get()
                results = self.model.predict(source=frame, imgsz=imgsz, conf=conf, stream=False)[0]
                annotated_frame = results.plot()

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.image = imgtk

                if self.save_var.get() and out:
                    out.write(annotated_frame)

                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

                num_objects = len(results.boxes)
                class_counts = results.boxes.cls.tolist()
                car_count = class_counts.count(0)
                bike_count = class_counts.count(1)

                self.update_metrics_text(f"Épocas Totais: N/A (ao vivo)\n"
                                        f"mAP@50: N/A (ao vivo)\n"
                                        f"mAP@50:95: N/A (ao vivo)\n"
                                        f"Precisão: N/A (ao vivo)\n"
                                        f"Recall: N/A (ao vivo)\n"
                                        f"FPS: {self.fps:.2f}\n"
                                        f"Objetos Detectados: {num_objects}\n"
                                        f"Carros: {car_count}\n"
                                        f"Bikes: {bike_count}")

                self.root.update_idletasks()
                self.status_var.set(f"Frame {int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))} de {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))} | FPS: {self.fps:.2f}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro durante detecção: {str(e)}")
                logging.error(f"Erro durante detecção: {str(e)}")
                self.running = False
                break

        if self.save_var.get() and out:
            out.release()
            self.status_var.set(f"Vídeo salvo em: {output_path}")
            logging.info(f"Vídeo salvo: {output_path}")
        elif self.running:
            self.status_var.set("Detecção concluída")
        self.stop_detection()

    def stop_detection(self):
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        self.canvas.delete("all")
        self.status_var.set("Pronto")
        self.load_training_metrics()
        logging.info("Detecção parada.")

    def save_results(self):
        if not os.path.exists(self.run_dir):
            messagebox.showerror("Erro", "Diretório de resultados não encontrado.")
            logging.error("Diretório de resultados não encontrado ao tentar salvar.")
            return

        save_dir = filedialog.askdirectory()
        if save_dir:
            try:
                shutil.copytree(self.run_dir, os.path.join(save_dir, f"run_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
                messagebox.showinfo("Sucesso", "Resultados salvos com sucesso!")
                logging.info(f"Resultados salvos em: {save_dir}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar resultados: {str(e)}")
                logging.error(f"Erro ao salvar resultados: {str(e)}")

    def show_dataset_images(self):
        dataset_dir = os.path.join(self.run_dir, "..", "dataset")
        image_files = glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True)[:5]

        if not image_files:
            messagebox.showinfo("Informação", "Nenhuma imagem encontrada no dataset.")
            logging.info("Nenhuma imagem encontrada no dataset.")
            return

        img_window = tk.Toplevel(self.root)
        img_window.title("Imagens do Dataset")
        img_window.geometry("800x600")

        canvas = tk.Canvas(img_window, bg="white")
        scrollbar = ttk.Scrollbar(img_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        for img_path in image_files:
            try:
                img = Image.open(img_path)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                label = ttk.Label(scrollable_frame, image=imgtk)
                label.image = imgtk
                label.pack(pady=5)
                ttk.Label(scrollable_frame, text=os.path.basename(img_path)).pack()
            except Exception as e:
                ttk.Label(scrollable_frame, text=f"Erro ao carregar {img_path}: {str(e)}").pack()
                logging.error(f"Erro ao carregar imagem do dataset: {str(e)}")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def on_closing(self):
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.root.destroy()
        logging.info("Aplicação fechada.")

def main():
    parser = argparse.ArgumentParser(description="Suite Avançada de Detecção com YOLOv8")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Caminho para o modelo treinado")
    args = parser.parse_args()

    root = tk.Tk()
    app = VideoDetectionApp(root, args.model)
    root.mainloop()

if __name__ == "__main__":
    main()