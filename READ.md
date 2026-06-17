# 🎭 Face Detection & Recognition System

This is an AI/ML computer vision project built entirely inside a **Jupyter Notebook**. The system uses a deep learning Convolutional Neural Network (CNN) architecture to detect human faces and recognize identities dynamically.

---

## 🧠 Tech Stack & Libraries Used
The project is built using Python and the following core data science and deep learning libraries:
- **TensorFlow & Keras:** For designing, compiling, and training the sequential CNN model layers.
- **OpenCV (opencv-contrib-python):** For image processing, cascade classifiers, and handling live camera feeds.
- **Pandas & NumPy:** For dataset structuring, matrix operations, and handling image arrays.
- **Tqdm:** For displaying smart, interactive progress bars during data preprocessing and loading.
- **Jupyter Notebook:** The interactive environment used for prototyping, visualizing data, and training.

---

## ⚙️ Features & Model Workflow
1. **Data Preprocessing:** Input images are converted to grayscale and resized to `48x48x1` dimensions.
2. **CNN Architecture:** Stacked layers of `Conv2D`, `MaxPooling2D`, and `Dropout` to extract spatial facial features without overfitting.
3. **Speed Optimization:** Configured with custom epoch settings optimized for local CPU execution to prevent system freezing.
4. **Visual Progress Tracking:** Leverages `tqdm` loops to monitor deep learning image matrices loading in real-time.

---

## 📂 Project Structure
```text
Face_Detection_Project/
│
├── Face_Detection_Recognition.ipynb  # Main Jupyter Notebook with all the code
├── kaggle_face_data/                  # Organized dataset containing face folders
├── face_model.h5                     # Saved trained Keras model weights
├── system_monitor.py                 # Script to monitor local CPU/RAM loads
└── README.md                         # Project documentation
```

---

## 🛠️ How to Run the Project

1. **Clone this repository:**
   ```bash
   git clone https://github.com
   cd YOUR_REPO_NAME
   ```

2. **Install all required dependencies:**
   ```bash
   pip install tensorflow keras pandas numpy tqdm opencv-contrib-python psutil GPUtil
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open `Face_Detection_Recognition.ipynb` and run the cells sequentially (`Shift + Enter`).

---

## 🤝 Developer Profile
- **Developer Role:** Python & AI/ML Engineer 🤖💻
- **GitHub Profile:** [@YOUR_USERNAME](https://github.com)
