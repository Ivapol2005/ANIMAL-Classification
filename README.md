# ANIMAL-Classification: Deep Learning for Image Classification

## Overview

This project implements a robust deep learning system designed for classifying animal images (cats and dogs) by their specific breed. It leverages Convolutional Neural Networks (CNNs) to accurately identify 37 distinct animal breeds directly from images.

## Key Features

* **Breed Classification:** Accurately classifies images of cats and dogs into 37 distinct breeds, primarily utilizing the **Oxford-IIIT Pet Dataset**.
* **Deep Learning (CNN):** Employs a sophisticated Convolutional Neural Network (`cat_dog_classifier.h5`) for high-performance image recognition.
* **Object Detection Support:** Incorporates functionalities related to VOC dataset annotations, indicating a foundation for broader object detection capabilities beyond simple classification.
* **Streamlined Pipeline:** Provides an efficient, end-to-end automated pipeline for image classification.

## Technologies Used

This project is built using Python and leverages the following key libraries:

* **Deep Learning & Machine Learning:** `TensorFlow`, `Keras` (as part of TensorFlow), `scikit-learn`, `numpy`
* **Computer Vision:** `OpenCV` (`cv2`), `matplotlib`
* **Data Processing & Utilities:** `pandas` (if used for data handling), `os`, `xml.etree.ElementTree`, `imblearn` (if used for handling imbalanced datasets)

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.x
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ivapol2005/ANIMAL-Classification.git](https://github.com/Ivapol2005/ANIMAL-Classification.git)
    cd ANIMAL-Classification
    ```
2.  **(Optional, but Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows (Command Prompt):
    # .\venv\Scripts\activate.bat
    # On Windows (PowerShell):
    # .\venv\Scripts\Activate.ps1
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Project Structure

The project is organized into the following main directories and files:

ANIMAL-Classification/
├── .git/                      # Git version control directory
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── README.md                  # The main project documentation (this file)
├── LICENSE                    # Contains the licensing information for the project (e.g., MIT License)
├── requirements.txt           # Lists all Python package dependencies required for the project
├── animal_classification.ipynb # Jupyter Notebook for in-depth data analysis and model experimentation
├── dataset-analysis.ipynb     # Another Jupyter Notebook for exploring datasets and initial insights
├── animal_classification_CatDog.py # Script to launch the application.
├── code/                      # Main source code directory (where your Python scripts reside)
│   ├── init.py            # Marks 'code' as a Python package (essential for imports like from code.breeds_list import ...)
│   ├── breeds_list.py         # (Assuming this is inside your 'code' folder) Defines animal breed lists.
│   ├── create_model.py        # Script responsible for training and saving the Convolutional Neural Network (CNN) model.
│   ├── launch.py              # (If this is a main entry point or UI launcher) Script to launch the application.
│   ├── pipeline.py            # The core script that orchestrates the image classification pipeline.
│   └── testAI.py              # Contains scripts or functions for testing the AI functionalities.
├── data/                      # Directory for raw or processed datasets
│   └── test-animals/          # A subdirectory holding example test images for classification.
└── models/                    # Stores trained machine learning models
└── cat_dog_classifier.h5  # The pre-trained or newly trained CNN model for animal breed classification.


## Usage

This section outlines how to interact with the project's core functionalities.

### Running a Quick Classification Example

To perform a quick test classification on an image, you can use a simple Python script.

1.  **Ensure your `test-animals` directory is correctly placed within the `data/` folder.**
    If you haven't already, move it:
    ```bash
    mv test-animals data/
    ```
2.  **Create a new Python file** (e.g., `run_example.py`) in the root of your project `ANIMAL-Classification/` with the following content:

    ```python
    # ANIMAL-Classification/run_example.py
    import os
    import sys

    # Add the 'code' directory to Python's path to allow direct imports
    sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

    # Now you can import your module
    import animal_classification_CatDog

    # Define the path to your test image (assuming it's in data/test-animals/)
    image_path = "data/test-animals/Gustav_chocolate.jpg"

    # Load the image using your module's function
    image = animal_classification_CatDog.load_image_file(image_path)

    # Perform the breed classification
    result = animal_classification_CatDog.breed(image, image_path=image_path)

    # Print the result
    print(f"Classification Result for {image_path}:")
    print(result)
    ```

3.  **Run the example script:**
    ```bash
    python run_example.py
    ```
    ![Example of model useage](Figure 1.png)
