
# 🔬 Medical Image Processing App

An interactive **Streamlit-based application** for processing and analyzing medical images.  
The app covers a wide range of techniques—from basic image adjustments to advanced tomographic reconstruction.

---

## 🌟 Features

The application is organized into **four main modules**:

---

### 📊 **Chapter 1: Imaging Methods**

- Windowing techniques: **Simple**, **Broken**, **Double Window**  
- LUT Transformations: **Inverse**, **Logarithmic**, **Sigmoid**, and more  
- Histogram Equalization: **HE**, **CLAHE**

---

### 🔲 **Chapter 2: Spatial Domain Filtering**

- Smoothing filters (mean, Gaussian, median)  
- Edge detection: **Sobel**, **Laplacian**  
- Sharpening filters  
- Support for **custom kernels**  
- Noise analysis and pixel intensity profiling

---

### 〰️ **Chapter 3: Frequency Domain Filtering**

- Frequency filters: **Ideal**, **Butterworth**, **Gaussian**, **Exponential**  
- Image restoration using **Wiener** and **Power filters**  
- FFT magnitude spectrum visualization

---

### 🔁 **Chapter 4: Tomographic Reconstruction**

- Implementation of **Filtered Back Projection (FBP)**  
- Iterative reconstruction using **ART / SART**  
- Sinogram generation and visualization

---

## 🛠️ Technologies & Libraries

This project uses:

- **Streamlit** — User interface  
- **NumPy**, **SciPy** — Mathematical operations  
- **Matplotlib** — Visualization  
- **Pillow**, **pydicom** — Image & DICOM handling  
- **scikit-image** — Radon/Iradon transforms  
- **OpenCV (cv2)** — Image processing  

---

## 🚀 Local Installation

Clone the repository:

```bash
git clone https://github.com/Eleni-papameleti/Medical-Image-Processing-App.git
cd Medical-Image-Processing-App
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

