import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import random
import threading
import time

class VHS:
    def __init__(self, lumaCompressionRate=1.92, lumaNoiseSigma=2, lumaNoiseMean=-10,
                 chromaCompressionRate=16, chromaNoiseIntensity=2,
                 verticalBlur=3, horizontalBlur=1.7, borderSize=1.7):
        self.lumaCompressionRate = lumaCompressionRate
        self.lumaNoiseSigma = lumaNoiseSigma
        self.lumaNoiseMean = lumaNoiseMean
        self.chromaCompressionRate = chromaCompressionRate
        self.chromaNoiseIntensity = chromaNoiseIntensity
        self.verticalBlur = verticalBlur
        self.horizontalBlur = horizontalBlur
        self.borderSize = borderSize / 100
        self.generation = 3

    def addNoise(self, image, mean=0, sigma=30):
        height, width, channels = image.shape
        noisy_image = np.copy(image)
        gaussian_noise = np.random.normal(mean, sigma, (height, width, channels))
        noisy_image = np.clip(noisy_image + gaussian_noise, 0, 255).astype(np.uint8)
        return noisy_image

    def addChromaNoise(self, image, intensity=10):
        height, width = image.shape[:2]
        noise_red = np.random.randint(-intensity, intensity, (height, width), dtype=np.int16)
        noise_green = np.random.randint(-intensity, intensity, (height, width), dtype=np.int16)
        noise_blue = np.random.randint(-intensity, intensity, (height, width), dtype=np.int16)
        image[:, :, 0] = np.clip(image[:, :, 0] + noise_blue, 0, 255)
        image[:, :, 1] = np.clip(image[:, :, 1] + noise_green, 0, 255)
        image[:, :, 2] = np.clip(image[:, :, 2] + noise_red, 0, 255)
        image = np.uint8(image)
        return image

    def cut_black_line_border(self, image: np.ndarray, bordersize: int = None) -> None:
        h, w, _ = image.shape
        if bordersize is None:
            line_width = int(w * self.borderSize)  # 1.7%
        else:
            line_width = bordersize
        image[:, -1 * line_width:] = 0

    def compressLuma(self, image):
        height, width = image.shape[:2]
        step1 = cv2.resize(image, (int(width / self.lumaCompressionRate), int(height)), interpolation=cv2.INTER_LANCZOS4)
        step1 = self.addNoise(step1, self.lumaNoiseMean, self.lumaNoiseSigma)
        step2 = cv2.resize(step1, (width, height), interpolation=cv2.INTER_LANCZOS4)
        self.cut_black_line_border(step2)
        return step2

    def compressChroma(self, image):
        height, width = image.shape[:2]
        step1 = cv2.resize(image, (int(width / self.chromaCompressionRate), int(height)), interpolation=cv2.INTER_LANCZOS4)
        step1 = self.addChromaNoise(step1, self.chromaNoiseIntensity)
        step2 = cv2.resize(step1, (width, height), interpolation=cv2.INTER_LANCZOS4)
        self.cut_black_line_border(step2)
        return step2

    def blur(self, image):
        filtered_image = cv2.blur(image, (int(self.horizontalBlur), int(self.verticalBlur)))
        return filtered_image

    def waves(self, img):
        rows, cols = img.shape[:2]
        i, j = np.indices((rows, cols))
        waves = round(random.uniform(0.000, 1.110), 3)
        offset_x = (waves * np.sin(250 * 2 * np.pi * i / (2 * cols))).astype(int)
        offset_j = j + offset_x
        offset_j = np.clip(offset_j, 0, cols - 1)
        img_output = img[i, offset_j]
        return img_output

    def waves2(self, img):
        rows, cols = img.shape[:2]
        i, j = np.indices((rows, cols))
        waves = round(random.uniform(1.000, 1.110), 3)
        offset_x = ((waves * np.sin(np.cos(random.randint(200, 250)) * 2 * np.pi * i / (2 * cols)))).astype(int)
        offset_j = j + offset_x
        offset_j = np.clip(offset_j, 0, cols - 1)
        img_output = img[i, offset_j]
        return img_output

    def switchNoise(self, img):
        rows, cols = img.shape[:2]
        i, j = np.indices((rows, cols))
        waves = round(random.uniform(1.900, 1.910), 3)
        offset_x = (waves * np.sin(np.cos(250) * 2 * np.pi * i / (2 * cols))).astype(int)
        offset_j = j + (offset_x * random.randint(20, 30))
        offset_j = np.clip(offset_j, 0, cols - 1)
        img_output = img[i, offset_j]
        return img_output

    def sharpen2(self, image, kernel_size=(5, 5), sigma=100, alpha=1.5, beta=-0.5):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        unsharp_mask = cv2.addWeighted(image, alpha, blurred, beta, 0)
        unsharp_mask = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        return unsharp_mask

    def processFrame(self, image):
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        luma_compressed = self.compressLuma(image_ycrcb)
        chroma_compressed = self.compressChroma(image_ycrcb)
        chroma_compressed = self.waves(chroma_compressed)
        chroma_compressed = self.waves(chroma_compressed)
        chroma_compressed = cv2.medianBlur(chroma_compressed, 1)
        chrominance_layer = chroma_compressed[:, :, 1:3]
        merged_ycrcb = cv2.merge([luma_compressed[:, :, 0], chrominance_layer])
        chrominance_bgr = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)
        height, width, _ = chrominance_bgr.shape
        stripe_width = int(width * self.borderSize)
        chrominance_bgr[:, -stripe_width:, 1] = 0
        return chrominance_bgr

    def sharpen_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        hpf = image - cv2.GaussianBlur(image, (21, 21), 3) + 127
        image_norm = image.astype(float) / 255.0
        hpf_norm = hpf.astype(float) / 255.0
        blended_norm = np.where(hpf_norm <= 0.5,
                                2 * image_norm * hpf_norm,
                                1 - 2 * (1 - image_norm) * (1 - hpf_norm))
        blended_norm = np.clip(blended_norm, 0, 1)
        blended = (blended_norm * 255).astype(np.uint8)
        return blended

    def processAll(self, image):
        image = self.sharpen_image(image)
        image = self.switchNoise(image)
        image = self.processFrame(image)
        image = self.waves(image)
        image = self.waves2(image)
        image = self.blur(image)
        image = self.sharpen2(image)
        return image

    def applyVHSEffect(self, image):
        originalValues = [self.lumaNoiseSigma, self.lumaNoiseMean, self.chromaNoiseIntensity]
        self.lumaNoiseSigma *= self.generation
        self.lumaNoiseMean *= self.generation
        self.chromaNoiseIntensity *= self.generation
        image = self.processAll(image)
        self.lumaNoiseSigma, self.lumaNoiseMean, self.chromaNoiseIntensity = originalValues
        return image

def cv2_to_pil(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_img)

def pil_to_cv2(pil_img):
    open_cv_image = np.array(pil_img)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VHS Эффект - Фото и Видео с живым предпросмотром")
        self.root.geometry("1024x768")
        self.root.iconbitmap("sys\\icon.ico")

        self.vhs = VHS()

        # Верхняя надпись
        self.img_label = tk.Label(root, text="Загрузите фото или видео", font=("Arial", 14))
        self.img_label.pack(pady=5)

        # Показ изображения/кадра видео
        self.image_panel = tk.Label(root)
        self.image_panel.pack(pady=5)

        # Кнопки загрузки
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.btn_load_img = tk.Button(btn_frame, text="Загрузить фото", command=self.load_image)
        self.btn_load_img.grid(row=0, column=0, padx=10)

        self.btn_load_vid = tk.Button(btn_frame, text="Загрузить видео", command=self.load_video)
        self.btn_load_vid.grid(row=0, column=1, padx=10)

        self.btn_save = tk.Button(btn_frame, text="Применить и сохранить", command=self.apply_and_save, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=2, padx=10)

        # Пауза для видео
        self.btn_pause = tk.Button(btn_frame, text="Пауза видео", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.grid(row=0, column=3, padx=10)

        # Настройки слайдеры
        self.params_frame = tk.LabelFrame(root, text="Настройки VHS", padx=10, pady=10)
        self.params_frame.pack(padx=10, pady=10, fill="x")

        self.lumaCompressionRate_var = tk.DoubleVar(value=1.92)
        self.lumaNoiseSigma_var = tk.DoubleVar(value=2)
        self.lumaNoiseMean_var = tk.DoubleVar(value=-10)
        self.chromaCompressionRate_var = tk.DoubleVar(value=16)
        self.chromaNoiseIntensity_var = tk.DoubleVar(value=2)
        self.verticalBlur_var = tk.DoubleVar(value=3)
        self.horizontalBlur_var = tk.DoubleVar(value=1.7)
        self.borderSize_var = tk.DoubleVar(value=1.7)

        self.create_slider("lumaCompressionRate", self.lumaCompressionRate_var, 1, 5, 0.01)
        self.create_slider("lumaNoiseSigma", self.lumaNoiseSigma_var, 0, 50, 0.1)
        self.create_slider("lumaNoiseMean", self.lumaNoiseMean_var, -50, 50, 0.1)
        self.create_slider("chromaCompressionRate", self.chromaCompressionRate_var, 1, 50, 0.1)
        self.create_slider("chromaNoiseIntensity", self.chromaNoiseIntensity_var, 0, 50, 0.1)
        self.create_slider("verticalBlur", self.verticalBlur_var, 1, 10, 0.1)
        self.create_slider("horizontalBlur", self.horizontalBlur_var, 1, 10, 0.1)
        self.create_slider("borderSize (%)", self.borderSize_var, 0, 10, 0.1)

        # Привязка обновления при движении слайдеров
        for var in [self.lumaCompressionRate_var, self.lumaNoiseSigma_var, self.lumaNoiseMean_var,
                    self.chromaCompressionRate_var, self.chromaNoiseIntensity_var,
                    self.verticalBlur_var, self.horizontalBlur_var, self.borderSize_var]:
            var.trace_add('write', self.on_slider_change)

        self.loaded_image_cv2 = None
        self.processed_image_cv2 = None

        # Видео
        self.video_path = None
        self.cap = None
        self.is_video = False
        self.is_paused = False
        self.stop_video = False

        # Для плавного изменения параметров видео (слегка случайные глитчи)
        self.base_params = {}

    def create_slider(self, label, var, frm, to, resolution):
        frame = tk.Frame(self.params_frame)
        frame.pack(fill="x", pady=5)
        lbl = tk.Label(frame, text=label, width=20, anchor="w")
        lbl.pack(side="left")
        slider = tk.Scale(frame, variable=var, from_=frm, to=to, resolution=resolution, orient=tk.HORIZONTAL, length=400)
        slider.pack(side="right")

    def load_image(self):
        self.stop_video_playback()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not file_path:
            return
        pil_img = Image.open(file_path).convert("RGB")
        self.loaded_image_cv2 = pil_to_cv2(pil_img)
        self.is_video = False
        self.img_label.config(text=f"Фото загружено: {file_path.split('/')[-1]}")
        self.btn_save.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.update_preview()

    def load_video(self):
        self.stop_video_playback()
        file_path = filedialog.askopenfilename(filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv")])
        if not file_path:
            return
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео")
            return
        self.is_video = True
        self.is_paused = False
        self.stop_video = False
        self.img_label.config(text=f"Видео загружено: {file_path.split('/')[-1]}")
        self.btn_save.config(state=tk.DISABLED)  # сохранять видео пока не реализовано
        self.btn_pause.config(state=tk.NORMAL, text="Пауза видео")
        self.base_params = {
            'lumaCompressionRate': self.lumaCompressionRate_var.get(),
            'lumaNoiseSigma': self.lumaNoiseSigma_var.get(),
            'lumaNoiseMean': self.lumaNoiseMean_var.get(),
            'chromaCompressionRate': self.chromaCompressionRate_var.get(),
            'chromaNoiseIntensity': self.chromaNoiseIntensity_var.get(),
            'verticalBlur': self.verticalBlur_var.get(),
            'horizontalBlur': self.horizontalBlur_var.get(),
            'borderSize': self.borderSize_var.get()
        }
        threading.Thread(target=self.play_video, daemon=True).start()

    def stop_video_playback(self):
        self.stop_video = True
        self.is_video = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def play_video(self):
        while not self.stop_video and self.cap.isOpened():
            if self.is_paused:
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                # Видео закончилось
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Применяем VHS эффект с небольшими глитч-колебаниями параметров
            vhs_params = {}
            for key, base_val in self.base_params.items():
                # колебания ±5% с плавным шумом
                jitter = base_val * 0.05 * random.uniform(-1, 1)
                vhs_params[key] = max(0.01, base_val + jitter)

            vhs = VHS(**vhs_params)
            processed = vhs.applyVHSEffect(frame)

            # Показать в UI (в main thread через after)
            self.processed_image_cv2 = processed
            self.root.after(0, self.update_image_panel, processed)

            # Фреймрейт ~30 FPS
            time.sleep(1/30)

    def update_image_panel(self, cv_img):
        pil_img = cv2_to_pil(cv_img)
        # масштабируем под окно (макс 800x450)
        pil_img.thumbnail((800, 450))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_panel.configure(image=tk_img)
        self.image_panel.image = tk_img

    def apply_and_save(self):
        if self.loaded_image_cv2 is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите фото")
            return
        # применить VHS с текущими настройками и сохранить
        vhs = VHS(
            lumaCompressionRate=self.lumaCompressionRate_var.get(),
            lumaNoiseSigma=self.lumaNoiseSigma_var.get(),
            lumaNoiseMean=self.lumaNoiseMean_var.get(),
            chromaCompressionRate=self.chromaCompressionRate_var.get(),
            chromaNoiseIntensity=self.chromaNoiseIntensity_var.get(),
            verticalBlur=self.verticalBlur_var.get(),
            horizontalBlur=self.horizontalBlur_var.get(),
            borderSize=self.borderSize_var.get(),
        )
        result = vhs.applyVHSEffect(self.loaded_image_cv2)
        cv2.imwrite("output.png", result)
        messagebox.showinfo("Готово", "Фото сохранено как output.png")

    def on_slider_change(self, *args):
        if self.is_video:
            # Настройки изменились — обновим базовые параметры для видео
            self.base_params = {
                'lumaCompressionRate': self.lumaCompressionRate_var.get(),
                'lumaNoiseSigma': self.lumaNoiseSigma_var.get(),
                'lumaNoiseMean': self.lumaNoiseMean_var.get(),
                'chromaCompressionRate': self.chromaCompressionRate_var.get(),
                'chromaNoiseIntensity': self.chromaNoiseIntensity_var.get(),
                'verticalBlur': self.verticalBlur_var.get(),
                'horizontalBlur': self.horizontalBlur_var.get(),
                'borderSize': self.borderSize_var.get()
            }
            return
        if self.loaded_image_cv2 is not None:
            self.update_preview()

    def update_preview(self):
        # Применяем VHS к текущему фото с текущими настройками без сохранения
        vhs = VHS(
            lumaCompressionRate=self.lumaCompressionRate_var.get(),
            lumaNoiseSigma=self.lumaNoiseSigma_var.get(),
            lumaNoiseMean=self.lumaNoiseMean_var.get(),
            chromaCompressionRate=self.chromaCompressionRate_var.get(),
            chromaNoiseIntensity=self.chromaNoiseIntensity_var.get(),
            verticalBlur=self.verticalBlur_var.get(),
            horizontalBlur=self.horizontalBlur_var.get(),
            borderSize=self.borderSize_var.get(),
        )
        try:
            processed = vhs.applyVHSEffect(self.loaded_image_cv2)
            self.processed_image_cv2 = processed
            self.update_image_panel(processed)
        except Exception as e:
            print(f"Ошибка обработки фото: {e}")

    def toggle_pause(self):
        if not self.is_video:
            return
        self.is_paused = not self.is_paused
        self.btn_pause.config(text="Продолжить видео" if self.is_paused else "Пауза видео")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
