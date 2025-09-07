import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import random
import threading
import time
import os
from datetime import datetime
import queue

class VHS:
    def __init__(self, lumaCompressionRate=1.92, lumaNoiseSigma=2, lumaNoiseMean=-10,
                 chromaCompressionRate=16, chromaNoiseIntensity=2,
                 verticalBlur=3, horizontalBlur=1.7, borderSize=1.7):
        self.lumaCompressionRate = lumaCompressionRate
        self.lumaNoiseSigma = lumaNoiseSigma
        self.lumaNoiseMean = lumaNoiseMean
        self.chromaCompressionRate = chromaCompressionRate
        self.chromaNoiseIntensity = chromaNoiseIntensity
        self.verticalBlur = max(1, int(verticalBlur))
        self.horizontalBlur = max(1, int(horizontalBlur))
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
            line_width = int(w * self.borderSize)
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
        # Гарантируем целые числа и минимальный размер 1
        ksize_x = max(1, int(self.horizontalBlur))
        ksize_y = max(1, int(self.verticalBlur))
        # Делаем размер ядра нечетным
        ksize_x = ksize_x if ksize_x % 2 == 1 else ksize_x + 1
        ksize_y = ksize_y if ksize_y % 2 == 1 else ksize_y + 1
        filtered_image = cv2.GaussianBlur(image, (ksize_x, ksize_y), 0)
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

class VideoRenderer:
    def __init__(self, input_path, output_path, vhs_params, progress_callback=None):
        self.input_path = input_path
        self.output_path = output_path
        self.vhs_params = vhs_params
        self.progress_callback = progress_callback
        self.stop_rendering = False
        
        # Получаем информацию о видео
        self.cap = cv2.VideoCapture(input_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Настройки кодека для быстрого кодирования
        if output_path.endswith('.mp4'):
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        self.cap.release()

    def render(self):
        try:
            # Создаем VideoWriter с настройками для быстрой записи
            out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
            
            if not out.isOpened():
                return False, "Не удалось создать выходной файл"
            
            # Создаем пул потоков для параллельной обработки
            frame_queue = queue.Queue(maxsize=10)
            processed_queue = queue.Queue(maxsize=10)
            
            # Запускаем потоки для обработки кадров
            processing_threads = []
            for _ in range(4):  # 4 потока для обработки
                thread = threading.Thread(target=self._process_frames_worker, 
                                        args=(frame_queue, processed_queue))
                thread.daemon = True
                thread.start()
                processing_threads.append(thread)
            
            # Поток для записи кадров
            write_thread = threading.Thread(target=self._write_frames_worker, 
                                          args=(out, processed_queue))
            write_thread.daemon = True
            write_thread.start()
            
            # Читаем и отправляем кадры на обработку
            cap = cv2.VideoCapture(self.input_path)
            current_frame = 0
            
            while not self.stop_rendering and current_frame < self.total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_queue.put((current_frame, frame))
                current_frame += 1
                
                if self.progress_callback and current_frame % 10 == 0:
                    progress = (current_frame / self.total_frames) * 100
                    self.progress_callback(progress, current_frame)
            
            # Сигнализируем о завершении
            for _ in range(len(processing_threads)):
                frame_queue.put((None, None))
            
            # Ждем завершения обработки
            for thread in processing_threads:
                thread.join(timeout=1.0)
            
            # Сигнализируем о завершении записи
            processed_queue.put((None, None))
            write_thread.join(timeout=2.0)
            
            cap.release()
            out.release()
            
            return not self.stop_rendering, "Успешно" if not self.stop_rendering else "Прервано"
            
        except Exception as e:
            return False, f"Ошибка: {str(e)}"

    def _process_frames_worker(self, input_queue, output_queue):
        vhs = VHS(**self.vhs_params)
        
        while not self.stop_rendering:
            try:
                frame_data = input_queue.get(timeout=1.0)
                if frame_data[0] is None:
                    break
                
                frame_idx, frame = frame_data
                processed_frame = vhs.applyVHSEffect(frame)
                output_queue.put((frame_idx, processed_frame))
                
            except queue.Empty:
                if self.stop_rendering:
                    break
                continue

    def _write_frames_worker(self, out, input_queue):
        expected_frame = 0
        buffer = {}
        
        while not self.stop_rendering:
            try:
                frame_data = input_queue.get(timeout=1.0)
                if frame_data[0] is None:
                    break
                
                frame_idx, frame = frame_data
                buffer[frame_idx] = frame
                
                # Записываем кадры по порядку
                while expected_frame in buffer:
                    out.write(buffer[expected_frame])
                    del buffer[expected_frame]
                    expected_frame += 1
                    
            except queue.Empty:
                if self.stop_rendering:
                    break
                continue

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VHS Эффект - Фото и Видео с живым предпросмотром")
        self.root.geometry("1024x768")
        
        # Создаем папку для временных файлов
        self.temp_dir = "temp_vhs"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

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
        self.btn_load_img.grid(row=0, column=0, padx=5)

        self.btn_load_vid = tk.Button(btn_frame, text="Загрузить видео", command=self.load_video)
        self.btn_load_vid.grid(row=0, column=1, padx=5)

        self.btn_save = tk.Button(btn_frame, text="Сохранить фото", command=self.apply_and_save, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=2, padx=5)

        # Пауза для видео
        self.btn_pause = tk.Button(btn_frame, text="Пауза видео", command=self.toggle_pause, state=tk.DISABLED)
        self.btn_pause.grid(row=0, column=3, padx=5)

        # Рендеринг видео
        self.btn_render = tk.Button(btn_frame, text="Рендерить видео", command=self.render_video, state=tk.DISABLED)
        self.btn_render.grid(row=0, column=4, padx=5)

        self.btn_stop_render = tk.Button(btn_frame, text="Стоп рендер", command=self.stop_rendering, state=tk.DISABLED)
        self.btn_stop_render.grid(row=0, column=5, padx=5)

        # Прогресс бар для рендеринга
        self.progress_frame = tk.Frame(root)
        self.progress_frame.pack(pady=5, fill="x", padx=20)
        
        self.progress_label = tk.Label(self.progress_frame, text="", font=("Arial", 10))
        self.progress_label.pack()
        
        self.progress = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(fill="x", pady=2)
        
        self.time_label = tk.Label(self.progress_frame, text="", font=("Arial", 9))
        self.time_label.pack()
        
        self.progress_frame.pack_forget()

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
        self.video_fps = 30
        self.video_width = 0
        self.video_height = 0
        self.total_frames = 0

        # Для плавного изменения параметров видео
        self.base_params = {}

        # Для рендеринга
        self.is_rendering = False
        self.stop_render_flag = False
        self.renderer = None
        self.start_time = 0

    def create_slider(self, label, var, frm, to, resolution):
        frame = tk.Frame(self.params_frame)
        frame.pack(fill="x", pady=2)
        lbl = tk.Label(frame, text=label, width=20, anchor="w")
        lbl.pack(side="left")
        slider = tk.Scale(frame, variable=var, from_=frm, to=to, resolution=resolution, 
                         orient=tk.HORIZONTAL, length=400, showvalue=1)
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
        self.btn_render.config(state=tk.DISABLED)
        self.update_preview()

    def load_video(self):
        self.stop_video_playback()
        file_path = filedialog.askopenfilename(filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv")])
        if not file_path:
            return
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео")
            return
        
        # Получаем информацию о видео
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.is_video = True
        self.is_paused = False
        self.stop_video = False
        self.img_label.config(text=f"Видео загружено: {file_path.split('/')[-1]}")
        self.btn_save.config(state=tk.DISABLED)
        self.btn_render.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.NORMAL, text="Пауза видео")
        self.base_params = self.get_current_params()
        threading.Thread(target=self.play_video, daemon=True).start()

    def get_current_params(self):
        return {
            'lumaCompressionRate': self.lumaCompressionRate_var.get(),
            'lumaNoiseSigma': self.lumaNoiseSigma_var.get(),
            'lumaNoiseMean': self.lumaNoiseMean_var.get(),
            'chromaCompressionRate': self.chromaCompressionRate_var.get(),
            'chromaNoiseIntensity': self.chromaNoiseIntensity_var.get(),
            'verticalBlur': self.verticalBlur_var.get(),
            'horizontalBlur': self.horizontalBlur_var.get(),
            'borderSize': self.borderSize_var.get()
        }

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
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Применяем VHS эффект с текущими параметрами
            vhs = VHS(**self.base_params)
            try:
                processed = vhs.applyVHSEffect(frame)
                self.processed_image_cv2 = processed
                self.root.after(0, self.update_image_panel, processed)
            except Exception as e:
                print(f"Ошибка обработки кадра: {e}")
                continue

            time.sleep(1/30)

    def update_image_panel(self, cv_img):
        pil_img = cv2_to_pil(cv_img)
        pil_img.thumbnail((800, 450))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_panel.configure(image=tk_img)
        self.image_panel.image = tk_img

    def apply_and_save(self):
        if self.loaded_image_cv2 is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите фото")
            return
        
        vhs = VHS(**self.get_current_params())
        result = vhs.applyVHSEffect(self.loaded_image_cv2)
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, result)
            messagebox.showinfo("Готово", f"Фото сохранено как {file_path}")

    def render_video(self):
        if not self.is_video or self.video_path is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите видео")
            return
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if not output_path:
            return
        
        # Показываем прогресс бар
        self.progress_frame.pack(pady=5, fill="x", padx=20)
        self.progress['value'] = 0
        self.progress_label.config(text="Подготовка к рендерингу...")
        self.time_label.config(text="")
        
        self.is_rendering = True
        self.stop_render_flag = False
        self.btn_render.config(state=tk.DISABLED)
        self.btn_stop_render.config(state=tk.NORMAL)
        self.start_time = time.time()
        
        # Запускаем рендеринг в отдельном потоке
        threading.Thread(target=self._render_thread, args=(output_path,), daemon=True).start()

    def _render_thread(self, output_path):
        vhs_params = self.get_current_params()
        self.renderer = VideoRenderer(self.video_path, output_path, vhs_params, self._update_render_progress)
        
        success, message = self.renderer.render()
        
        self.root.after(0, self._finish_rendering, success, message, output_path)

    def _update_render_progress(self, progress, current_frame):
        if self.stop_render_flag:
            self.renderer.stop_rendering = True
            return
        
        elapsed = time.time() - self.start_time
        if progress > 0:
            remaining = (elapsed / progress) * (100 - progress)
            time_text = f"Прошло: {elapsed:.1f}с | Осталось: {remaining:.1f}с"
        else:
            time_text = f"Прошло: {elapsed:.1f}с"
        
        self.root.after(0, lambda: self.progress_label.config(
            text=f"Рендеринг: {progress:.1f}% ({current_frame}/{self.total_frames} кадров)"))
        self.root.after(0, lambda: self.progress.config(value=progress))
        self.root.after(0, lambda: self.time_label.config(text=time_text))

    def _finish_rendering(self, success, message, output_path):
        self.is_rendering = False
        self.progress_frame.pack_forget()
        self.btn_render.config(state=tk.NORMAL)
        self.btn_stop_render.config(state=tk.DISABLED)
        
        if success:
            messagebox.showinfo("Готово", f"Видео сохранено как {output_path}\n{message}")
        else:
            messagebox.showinfo("Завершено", message)

    def stop_rendering(self):
        if self.is_rendering:
            self.stop_render_flag = True
            if self.renderer:
                self.renderer.stop_rendering = True

    def on_slider_change(self, *args):
        if self.is_video:
            self.base_params = self.get_current_params()
        elif self.loaded_image_cv2 is not None:
            self.update_preview()

    def update_preview(self):
        vhs = VHS(**self.get_current_params())
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

    def on_closing(self):
        self.stop_video_playback()
        self.stop_rendering()
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()