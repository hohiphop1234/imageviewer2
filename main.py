import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


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


class ImageViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Viewer with Filters")
        self.root.geometry("1280x800")

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
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self.root, padding=10)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.main_panel = ttk.Frame(self.root, padding=10)
        self.main_panel.grid(row=0, column=1, sticky="nsew")
        self.main_panel.columnconfigure(0, weight=1)
        self.main_panel.rowconfigure(0, weight=1)

        self._build_menu()
        self._build_sidebar()
        self._build_image_display()
        self._build_transform_panel()

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.open_image)
        file_menu.add_command(label="Save As...", command=self.save_image, state="disabled")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        self.root.config(menu=menubar)
        self.file_menu = file_menu

    def _build_sidebar(self):
        notebook = ttk.Notebook(self.sidebar)
        notebook.pack(fill="both", expand=True)

        smooth_frame = ttk.Frame(notebook)
        notebook.add(smooth_frame, text="Smoothing")
        self._build_smoothing_filters(smooth_frame)

        statistical_frame = ttk.Frame(notebook)
        notebook.add(statistical_frame, text="Statistical")
        self._build_statistical_filters(statistical_frame)

        sharpening_frame = ttk.Frame(notebook)
        notebook.add(sharpening_frame, text="Sharpening")
        self._build_sharpening_filters(sharpening_frame)

    def _build_image_display(self):
        self.image_panel = ttk.LabelFrame(self.main_panel, text="Image Preview")
        self.image_panel.grid(row=0, column=0, sticky="nsew")
        self.image_panel.rowconfigure(0, weight=1)
        self.image_panel.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(self.image_panel, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        self.image_panel.bind("<Configure>", lambda event: self.display_image())

    def _build_transform_panel(self):
        panel = ttk.LabelFrame(self.main_panel, text="Intensity Transforms", padding=10)
        panel.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        panel.columnconfigure(1, weight=1)

        row = 0
        # Negative
        ttk.Checkbutton(panel, text="Negative", variable=self.transform_state.negative).grid(column=0, row=row, sticky="w")
        row += 1

        self._build_log_controls(panel, row)
        row += 4
        self._build_gamma_controls(panel, row)
        row += 4
        self._build_piecewise_controls(panel, row)
        row += 3

        button_row = ttk.Frame(panel)
        button_row.grid(column=0, row=row, columnspan=2, pady=(10, 0), sticky="ew")
        button_row.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(button_row, text="Update", command=self.apply_intensity_transforms).grid(column=0, row=0, sticky="ew", padx=5)
        ttk.Button(button_row, text="Save", command=self.save_image, state="disabled").grid(column=1, row=0, sticky="ew", padx=5)
        ttk.Button(button_row, text="Reset", command=self.reset_image).grid(column=2, row=0, sticky="ew", padx=5)

        self.update_button = button_row.grid_slaves(column=0, row=0)[0]
        self.save_button = button_row.grid_slaves(column=1, row=0)[0]

    def _build_log_controls(self, parent: ttk.Frame, start_row: int):
        ttk.Checkbutton(parent, text="Log (s = c·log_base(1 + r))", variable=self.transform_state.log_enabled).grid(column=0, row=start_row, sticky="w", pady=(10, 0))
        ttk.Label(parent, text="c").grid(column=0, row=start_row + 1, sticky="w")
        ttk.Scale(parent, from_=1, to=100, variable=self.transform_state.log_c, orient=tk.HORIZONTAL).grid(column=1, row=start_row + 1, sticky="ew", padx=5)
        ttk.Label(parent, text="base").grid(column=0, row=start_row + 2, sticky="w")
        ttk.Scale(parent, from_=2.0, to=10.0, variable=self.transform_state.log_base, orient=tk.HORIZONTAL).grid(column=1, row=start_row + 2, sticky="ew", padx=5)

    def _build_gamma_controls(self, parent: ttk.Frame, start_row: int):
        ttk.Checkbutton(parent, text="Gamma (s = c·r^γ)", variable=self.transform_state.gamma_enabled).grid(column=0, row=start_row, sticky="w", pady=(10, 0))
        ttk.Label(parent, text="c").grid(column=0, row=start_row + 1, sticky="w")
        ttk.Scale(parent, from_=0.1, to=3.0, variable=self.transform_state.gamma_c, orient=tk.HORIZONTAL).grid(column=1, row=start_row + 1, sticky="ew", padx=5)
        ttk.Label(parent, text="γ").grid(column=0, row=start_row + 2, sticky="w")
        ttk.Scale(parent, from_=0.1, to=5.0, variable=self.transform_state.gamma_gamma, orient=tk.HORIZONTAL).grid(column=1, row=start_row + 2, sticky="ew", padx=5)

    def _build_piecewise_controls(self, parent: ttk.Frame, start_row: int):
        ttk.Checkbutton(parent, text="Piecewise Stretch", variable=self.transform_state.piecewise_enabled).grid(column=0, row=start_row, sticky="w", pady=(10, 0))
        ttk.Label(parent, text="Low (r1)").grid(column=0, row=start_row + 1, sticky="w")
        low_scale = ttk.Scale(parent, from_=0, to=255, variable=self.transform_state.piecewise_low, orient=tk.HORIZONTAL)
        low_scale.grid(column=1, row=start_row + 1, sticky="ew", padx=5)
        ttk.Label(parent, text="High (r2)").grid(column=0, row=start_row + 2, sticky="w")
        high_scale = ttk.Scale(parent, from_=0, to=255, variable=self.transform_state.piecewise_high, orient=tk.HORIZONTAL)
        high_scale.grid(column=1, row=start_row + 2, sticky="ew", padx=5)

    # -------------------- Sidebar Sections -------------------- #
    def _build_section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.pack(fill="x", pady=5)
        for i in range(1):
            frame.columnconfigure(i, weight=1)
        return frame

    def _build_smoothing_filters(self, parent: ttk.Frame):
        section = self._build_section(parent, "Box Filter")
        for size in (3, 5, 7):
            ttk.Button(section, text=f"Box ({size}x{size})", command=lambda s=size: self.apply_filter(self.box_filter, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Gaussian Filter")
        for size in (3, 5, 7):
            ttk.Button(section, text=f"Gaussian ({size}x{size})", command=lambda s=size: self.apply_filter(self.gaussian_filter, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Lowpass Filter")
        ttk.Button(section, text="Apply Lowpass Filter", command=lambda: self.apply_filter(self.lowpass_filter, 3)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Median Filter")
        for size in (3, 5, 7):
            ttk.Button(section, text=f"Median ({size}x{size})", command=lambda s=size: self.apply_filter(self.median_filter, s)).pack(fill="x", pady=2)

    def _build_statistical_filters(self, parent: ttk.Frame):
        section = self._build_section(parent, "Max Filter")
        for size in (3, 5):
            ttk.Button(section, text=f"Max ({size}x{size})", command=lambda s=size: self.apply_filter(self.max_filter, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Min Filter")
        for size in (3, 5):
            ttk.Button(section, text=f"Min ({size}x{size})", command=lambda s=size: self.apply_filter(self.min_filter, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Max-Min Filter")
        for size in (3, 5):
            ttk.Button(section, text=f"Max-Min ({size}x{size})", command=lambda s=size: self.apply_filter(self.max_min_filter, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Midpoint Filter")
        for size in (3, 5):
            ttk.Button(section, text=f"Midpoint ({size}x{size})", command=lambda s=size: self.apply_filter(self.midpoint_filter, s)).pack(fill="x", pady=2)

    def _build_sharpening_filters(self, parent: ttk.Frame):
        section = self._build_section(parent, "Laplacian (edges)")
        for size in (3, 5):
            ttk.Button(section, text=f"Laplacian (edges {size}x{size})", command=lambda s=size: self.apply_filter(self.laplacian_edges, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Laplacian Sharp")
        for size in (3, 5):
            ttk.Button(section, text=f"Laplacian Sharp ({size}x{size})", command=lambda s=size: self.apply_filter(self.laplacian_sharp, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Unsharp Mask")
        for size in (3, 5, 7):
            ttk.Button(section, text=f"Unsharp Mask (k={size})", command=lambda s=size: self.apply_filter(self.unsharp_mask, s)).pack(fill="x", pady=2)

        section = self._build_section(parent, "Sobel Filter")
        ttk.Button(section, text="Sobel (magnitude)", command=lambda: self.apply_filter(self.sobel_magnitude)).pack(fill="x", pady=2)
        ttk.Button(section, text="Sobel X", command=lambda: self.apply_filter(self.sobel_axis, axis="x")).pack(fill="x", pady=2)
        ttk.Button(section, text="Sobel Y", command=lambda: self.apply_filter(self.sobel_axis, axis="y")).pack(fill="x", pady=2)

        section = self._build_section(parent, "Prewitt Filter")
        ttk.Button(section, text="Prewitt (magnitude)", command=lambda: self.apply_filter(self.prewitt_magnitude)).pack(fill="x", pady=2)
        ttk.Button(section, text="Prewitt X", command=lambda: self.apply_filter(self.prewitt_axis, axis="x")).pack(fill="x", pady=2)
        ttk.Button(section, text="Prewitt Y", command=lambda: self.apply_filter(self.prewitt_axis, axis="y")).pack(fill="x", pady=2)

        section = self._build_section(parent, "Roberts Filter")
        ttk.Button(section, text="Roberts (magnitude)", command=lambda: self.apply_filter(self.roberts_magnitude)).pack(fill="x", pady=2)

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
            self.image_label.config(text="Open an image to begin.")
            return

        max_width = self.image_panel.winfo_width() or 800
        max_height = self.image_panel.winfo_height() or 600
        image = self.processed_image.copy()
        image.thumbnail((max_width - 20, max_height - 20))
        self.display_image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.display_image_tk)

    def _update_buttons_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.save_button.config(state=state)
        self.file_menu.entryconfig("Save As...", state=state)

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
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

