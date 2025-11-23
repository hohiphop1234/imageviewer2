import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Set appearance mode and default color theme
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

@dataclass
class TransformState:
    negative: tk.BooleanVar
    log_enabled: tk.BooleanVar
    log_c: tk.DoubleVar
    log_base: tk.DoubleVar
    gamma_enabled: tk.BooleanVar
    gamma_c: tk.DoubleVar
    gamma_gamma: tk.DoubleVar
    piecewise_enabled: tk.BooleanVar
    piecewise_low: tk.IntVar
    piecewise_high: tk.IntVar


class ImageViewerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Viewer with Filters")
        self.geometry("1280x800")

        self.original_image: Optional[Image.Image] = None
        self.processed_image: Optional[Image.Image] = None
        self.display_image_tk: Optional[ImageTk.PhotoImage] = None

        self.transform_state = TransformState(
            negative=tk.BooleanVar(value=False),
            log_enabled=tk.BooleanVar(value=False),
            log_c=tk.DoubleVar(value=46.0),
            log_base=tk.DoubleVar(value=2.7),
            gamma_enabled=tk.BooleanVar(value=False),
            gamma_c=tk.DoubleVar(value=1.0),
            gamma_gamma=tk.DoubleVar(value=1.0),
            piecewise_enabled=tk.BooleanVar(value=False),
            piecewise_low=tk.IntVar(value=0),
            piecewise_high=tk.IntVar(value=255),
        )

        self._build_layout()

    # -------------------- UI Construction -------------------- #
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Image Filters", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # File Controls in Sidebar
        self.open_btn = ctk.CTkButton(self.sidebar, text="Open Image", command=self.open_image)
        self.open_btn.grid(row=1, column=0, padx=20, pady=10)
        
        self.save_btn = ctk.CTkButton(self.sidebar, text="Save Image", command=self.save_image, state="disabled")
        self.save_btn.grid(row=2, column=0, padx=20, pady=10)

        self.reset_btn = ctk.CTkButton(self.sidebar, text="Reset All", command=self.reset_image, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.reset_btn.grid(row=3, column=0, padx=20, pady=10)

        # Tabview for Filters
        self.tabview = ctk.CTkTabview(self.sidebar, width=250)
        self.tabview.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="nsew")
        
        self.tabview.add("Smooth")
        self.tabview.add("Stats")
        self.tabview.add("Sharp")
        self.tabview.add("Freq")
        self.tabview.add("Trans") # Intensity Transforms

        self._build_smoothing_filters(self.tabview.tab("Smooth"))
        self._build_statistical_filters(self.tabview.tab("Stats"))
        self._build_sharpening_filters(self.tabview.tab("Sharp"))
        self._build_fft_filters(self.tabview.tab("Freq"))
        self._build_transform_panel(self.tabview.tab("Trans"))

        # Main Area
        self.main_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_panel.grid(row=0, column=1, sticky="nsew")
        self.main_panel.grid_columnconfigure(0, weight=1)
        self.main_panel.grid_rowconfigure(0, weight=1)

        self._build_image_display()

    def _build_image_display(self):
        self.image_frame = ctk.CTkFrame(self.main_panel, corner_radius=10)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.image_frame, text="Open an image to begin.", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_frame.bind("<Configure>", lambda event: self.display_image())

    # -------------------- Sidebar Sections -------------------- #
    def _create_filter_btn(self, parent, text, command):
        btn = ctk.CTkButton(parent, text=text, command=command, height=30)
        btn.pack(fill="x", pady=5, padx=5)
        return btn

    def _build_smoothing_filters(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        
        ctk.CTkLabel(scroll, text="Box Filter", font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
        for size in (3, 5, 7):
            self._create_filter_btn(scroll, f"Box ({size}x{size})", lambda s=size: self.apply_filter(self.box_filter, s))

        ctk.CTkLabel(scroll, text="Gaussian Filter", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        for size in (3, 5, 7):
            self._create_filter_btn(scroll, f"Gaussian ({size}x{size})", lambda s=size: self.apply_filter(self.gaussian_filter, s))

        ctk.CTkLabel(scroll, text="Other", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        self._create_filter_btn(scroll, "Lowpass (3x3)", lambda: self.apply_filter(self.lowpass_filter, 3))
        
        ctk.CTkLabel(scroll, text="Median Filter", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        for size in (3, 5, 7):
            self._create_filter_btn(scroll, f"Median ({size}x{size})", lambda s=size: self.apply_filter(self.median_filter, s))

    def _build_statistical_filters(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll, text="Max Filter", font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
        for size in (3, 5):
            self._create_filter_btn(scroll, f"Max ({size}x{size})", lambda s=size: self.apply_filter(self.max_filter, s))

        ctk.CTkLabel(scroll, text="Min Filter", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        for size in (3, 5):
            self._create_filter_btn(scroll, f"Min ({size}x{size})", lambda s=size: self.apply_filter(self.min_filter, s))

        ctk.CTkLabel(scroll, text="Max-Min Filter", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        for size in (3, 5):
            self._create_filter_btn(scroll, f"Max-Min ({size}x{size})", lambda s=size: self.apply_filter(self.max_min_filter, s))

        ctk.CTkLabel(scroll, text="Midpoint Filter", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        for size in (3, 5):
            self._create_filter_btn(scroll, f"Midpoint ({size}x{size})", lambda s=size: self.apply_filter(self.midpoint_filter, s))

    def _build_sharpening_filters(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll, text="Laplacian", font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
        for size in (3, 5):
            self._create_filter_btn(scroll, f"Edges ({size}x{size})", lambda s=size: self.apply_filter(self.laplacian_edges, s))
        for size in (3, 5):
            self._create_filter_btn(scroll, f"Sharp ({size}x{size})", lambda s=size: self.apply_filter(self.laplacian_sharp, s))

        ctk.CTkLabel(scroll, text="Unsharp Mask", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        for size in (3, 5, 7):
            self._create_filter_btn(scroll, f"k={size}", lambda s=size: self.apply_filter(self.unsharp_mask, s))

        ctk.CTkLabel(scroll, text="Sobel", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        self._create_filter_btn(scroll, "Magnitude", lambda: self.apply_filter(self.sobel_magnitude))
        self._create_filter_btn(scroll, "X Axis", lambda: self.apply_filter(self.sobel_axis, axis="x"))
        self._create_filter_btn(scroll, "Y Axis", lambda: self.apply_filter(self.sobel_axis, axis="y"))

        ctk.CTkLabel(scroll, text="Prewitt", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        self._create_filter_btn(scroll, "Magnitude", lambda: self.apply_filter(self.prewitt_magnitude))
        self._create_filter_btn(scroll, "X Axis", lambda: self.apply_filter(self.prewitt_axis, axis="x"))
        self._create_filter_btn(scroll, "Y Axis", lambda: self.apply_filter(self.prewitt_axis, axis="y"))

        ctk.CTkLabel(scroll, text="Roberts", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        self._create_filter_btn(scroll, "Magnitude", lambda: self.apply_filter(self.roberts_magnitude))

    def _build_fft_filters(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll, text="Low Pass (FFT)", font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
        for radius in (20, 40, 60):
            self._create_filter_btn(scroll, f"Radius {radius}", lambda r=radius: self.apply_fft_filter(self.gaussian_lowpass_fft, r))

        ctk.CTkLabel(scroll, text="High Pass (FFT)", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0))
        for radius in (20, 40, 60):
            self._create_filter_btn(scroll, f"Radius {radius}", lambda r=radius: self.apply_fft_filter(self.gaussian_highpass_fft, r))

    def _build_transform_panel(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # Negative
        ctk.CTkCheckBox(scroll, text="Negative", variable=self.transform_state.negative).pack(anchor="w", pady=5)

        # Log
        ctk.CTkLabel(scroll, text="Log Transform", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0), anchor="w")
        ctk.CTkCheckBox(scroll, text="Enable Log", variable=self.transform_state.log_enabled).pack(anchor="w")
        
        ctk.CTkLabel(scroll, text="Constant (c)").pack(anchor="w")
        ctk.CTkSlider(scroll, from_=1, to=100, variable=self.transform_state.log_c).pack(fill="x", pady=2)
        
        ctk.CTkLabel(scroll, text="Base").pack(anchor="w")
        ctk.CTkSlider(scroll, from_=2.0, to=10.0, variable=self.transform_state.log_base).pack(fill="x", pady=2)

        # Gamma
        ctk.CTkLabel(scroll, text="Gamma Transform", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0), anchor="w")
        ctk.CTkCheckBox(scroll, text="Enable Gamma", variable=self.transform_state.gamma_enabled).pack(anchor="w")
        
        ctk.CTkLabel(scroll, text="Constant (c)").pack(anchor="w")
        ctk.CTkSlider(scroll, from_=0.1, to=3.0, variable=self.transform_state.gamma_c).pack(fill="x", pady=2)
        
        ctk.CTkLabel(scroll, text="Gamma (γ)").pack(anchor="w")
        ctk.CTkSlider(scroll, from_=0.1, to=5.0, variable=self.transform_state.gamma_gamma).pack(fill="x", pady=2)

        # Piecewise
        ctk.CTkLabel(scroll, text="Piecewise Stretch", font=ctk.CTkFont(weight="bold")).pack(pady=(10,0), anchor="w")
        ctk.CTkCheckBox(scroll, text="Enable Piecewise", variable=self.transform_state.piecewise_enabled).pack(anchor="w")
        
        ctk.CTkLabel(scroll, text="Low (r1)").pack(anchor="w")
        ctk.CTkSlider(scroll, from_=0, to=255, variable=self.transform_state.piecewise_low).pack(fill="x", pady=2)
        
        ctk.CTkLabel(scroll, text="High (r2)").pack(anchor="w")
        ctk.CTkSlider(scroll, from_=0, to=255, variable=self.transform_state.piecewise_high).pack(fill="x", pady=2)

        # Update Button
        ctk.CTkButton(scroll, text="Apply Transforms", command=self.apply_intensity_transforms, fg_color="green").pack(fill="x", pady=20)


    # -------------------- Image Management -------------------- #
    def open_image(self):
        filetypes = [("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Open image", filetypes=filetypes)
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return

        self.original_image = image
        self.processed_image = image.copy()
        self._update_buttons_state(enabled=True)
        self.display_image()

    def save_image(self):
        if self.processed_image is None:
            return
        filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tif")]
        path = filedialog.asksaveasfilename(title="Save image", defaultextension=".png", filetypes=filetypes)
        if not path:
            return
        try:
            self.processed_image.save(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not save image:\n{exc}")

    def reset_image(self):
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        self.transform_state.negative.set(False)
        self.transform_state.log_enabled.set(False)
        self.transform_state.gamma_enabled.set(False)
        self.transform_state.piecewise_enabled.set(False)
        self.transform_state.log_c.set(46.0)
        self.transform_state.log_base.set(2.7)
        self.transform_state.gamma_c.set(1.0)
        self.transform_state.gamma_gamma.set(1.0)
        self.transform_state.piecewise_low.set(0)
        self.transform_state.piecewise_high.set(255)
        self.display_image()

    def display_image(self):
        if self.processed_image is None:
            self.image_label.configure(text="Open an image to begin.", image=None)
            return

        # Calculate aspect ratio to fit in the frame
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        if frame_width < 10 or frame_height < 10:
            return # Too small to render yet

        image = self.processed_image.copy()
        img_w, img_h = image.size
        ratio = min(frame_width / img_w, frame_height / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.display_image_tk = ctk.CTkImage(light_image=image, dark_image=image, size=(new_w, new_h))
        self.image_label.configure(image=self.display_image_tk, text="")

    def _update_buttons_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.save_btn.configure(state=state)

    # -------------------- Filter Helper -------------------- #
    def apply_filter(self, filter_fn: Callable, *args, **kwargs):
        if self.processed_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        image_bgr = cv2.cvtColor(np.array(self.processed_image), cv2.COLOR_RGB2BGR)
        result_bgr = filter_fn(image_bgr, *args, **kwargs)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        self.processed_image = Image.fromarray(result_rgb)
        self.display_image()

    # -------------------- Filter Implementations -------------------- #
    @staticmethod
    def box_filter(image: np.ndarray, size: int) -> np.ndarray:
        return cv2.blur(image, (size, size))

    @staticmethod
    def gaussian_filter(image: np.ndarray, size: int) -> np.ndarray:
        return cv2.GaussianBlur(image, (size, size), 0)

    @staticmethod
    def lowpass_filter(image: np.ndarray, size: int) -> np.ndarray:
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def median_filter(image: np.ndarray, size: int) -> np.ndarray:
        return cv2.medianBlur(image, size)

    @staticmethod
    def max_filter(image: np.ndarray, size: int) -> np.ndarray:
        kernel = np.ones((size, size), np.uint8)
        return cv2.dilate(image, kernel)

    @staticmethod
    def min_filter(image: np.ndarray, size: int) -> np.ndarray:
        kernel = np.ones((size, size), np.uint8)
        return cv2.erode(image, kernel)

    def max_min_filter(self, image: np.ndarray, size: int) -> np.ndarray:
        max_img = self.max_filter(image, size)
        min_img = self.min_filter(image, size)
        diff = cv2.subtract(max_img, min_img)
        return diff

    def midpoint_filter(self, image: np.ndarray, size: int) -> np.ndarray:
        max_img = self.max_filter(image, size).astype(np.float32)
        min_img = self.min_filter(image, size).astype(np.float32)
        midpoint = ((max_img + min_img) / 2.0).astype(np.uint8)
        return midpoint

    @staticmethod
    def laplacian_edges(image: np.ndarray, size: int) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=size)
        lap = cv2.convertScaleAbs(lap)
        return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    def laplacian_sharp(self, image: np.ndarray, size: int) -> np.ndarray:
        lap = cv2.Laplacian(image, cv2.CV_16S, ksize=size)
        lap = cv2.convertScaleAbs(lap)
        sharp = cv2.addWeighted(image, 1.0, lap, -0.7, 0)
        return sharp

    def unsharp_mask(self, image: np.ndarray, size: int) -> np.ndarray:
        blur = cv2.GaussianBlur(image, (size, size), 0)
        mask = cv2.subtract(image, blur)
        sharpened = cv2.addWeighted(image, 1.0, mask, 1.0, 0)
        return sharpened

    @staticmethod
    def sobel_axis(image: np.ndarray, axis: str) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dx, dy = (1, 0) if axis == "x" else (0, 1)
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=3)
        abs_sobel = cv2.convertScaleAbs(sobel)
        return cv2.cvtColor(abs_sobel, cv2.COLOR_GRAY2BGR)

    def sobel_magnitude(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _prewitt_kernels():
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        return kernelx, kernely

    def prewitt_axis(self, image: np.ndarray, axis: str) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernelx, kernely = self._prewitt_kernels()
        kernel = kernelx if axis == "x" else kernely
        filtered = cv2.filter2D(gray, -1, kernel)
        abs_filtered = cv2.convertScaleAbs(filtered)
        return cv2.cvtColor(abs_filtered, cv2.COLOR_GRAY2BGR)

    def prewitt_magnitude(self, image: np.ndarray) -> np.ndarray:
        kernelx, kernely = self._prewitt_kernels()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.filter2D(gray, -1, kernelx)
        gy = cv2.filter2D(gray, -1, kernely)
        magnitude = cv2.magnitude(gx, gy)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    def roberts_magnitude(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray, -1, kernelx)
        gy = cv2.filter2D(gray, -1, kernely)
        magnitude = cv2.magnitude(gx, gy)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    # -------------------- FFT Implementation -------------------- #
    def apply_fft_filter(self, filter_fn: Callable, radius: int):
        if self.processed_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        # Convert to grayscale for FFT
        image_rgb = np.array(self.processed_image)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Pad to optimal size
        rows, cols = gray.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(gray, 0, nrows - rows, 0, ncols - cols, cv2.BORDER_CONSTANT, value=0)

        # DFT
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create mask
        mask = filter_fn(dft_shift.shape, radius)

        # Apply mask
        fshift = dft_shift * mask

        # Inverse DFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Crop back to original size
        img_back = img_back[:rows, :cols]

        # Normalize to 0-255
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        img_back = np.uint8(img_back)

        # Convert back to RGB (grayscale result)
        result_rgb = cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB)
        self.processed_image = Image.fromarray(result_rgb)
        self.display_image()

    @staticmethod
    def gaussian_lowpass_fft(shape, radius):
        rows, cols, _ = shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.float32)
        y, x = np.ogrid[:rows, :cols]
        # Gaussian formula: H(u,v) = exp(-D^2 / (2*D0^2))
        d_squared = (x - ccol) ** 2 + (y - crow) ** 2
        response = np.exp(-d_squared / (2 * (radius ** 2)))
        mask[:, :, 0] = response
        mask[:, :, 1] = response
        return mask

    @staticmethod
    def gaussian_highpass_fft(shape, radius):
        # HPF = 1 - LPF
        lpf_mask = ImageViewerApp.gaussian_lowpass_fft(shape, radius)
        return 1 - lpf_mask

    # -------------------- Intensity Transforms -------------------- #
    def apply_intensity_transforms(self):
        if self.processed_image is None:
            return
        image = np.array(self.processed_image, dtype=np.float32)

        if self.transform_state.negative.get():
            image = 255.0 - image

        if self.transform_state.log_enabled.get():
            c = self.transform_state.log_c.get()
            base = max(self.transform_state.log_base.get(), 1.01)
            image = c * (np.log1p(image) / math.log(base))
            image = np.clip(image, 0, 255)

        if self.transform_state.gamma_enabled.get():
            c = self.transform_state.gamma_c.get()
            gamma = self.transform_state.gamma_gamma.get()
            image_norm = np.clip(image / 255.0, 0, 1)
            image = c * np.power(image_norm, gamma) * 255.0
            image = np.clip(image, 0, 255)

        if self.transform_state.piecewise_enabled.get():
            low = self.transform_state.piecewise_low.get()
            high = self.transform_state.piecewise_high.get()
            if high <= low:
                messagebox.showwarning("Invalid range", "High (r2) must be greater than Low (r1).")
            else:
                image_norm = np.clip(image / 255.0, 0, 1)
                low_norm = low / 255.0
                high_norm = high / 255.0
                stretch = (image_norm - low_norm) / (high_norm - low_norm)
                stretch = np.clip(stretch, 0, 1)
                image = stretch * 255.0

        output = image.astype(np.uint8)
        self.processed_image = Image.fromarray(output)
        self.display_image()


def main():
    app = ImageViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
