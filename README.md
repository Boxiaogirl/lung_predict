
# CXR-RANet

## 1. Setup Instructions

### Enter the code file directory:
Use the `cd` command to navigate into the code directory.

### Create and activate Conda environment:

- **Create Conda environment:**
  ```bash
  conda create --name lung_predict python=3.8
  ```

- **Activate Conda environment:**
  ```bash
  conda activate lung_predict
  ```

### Install dependencies:
Install the required packages by running:
```bash
pip install -r ./requirements.txt
```

---

## 2. File Sharing

The following resources can be accessed through the shared link:

- **CXR-RANet Dataset, Pre-trained Model Files, and Code Packages**  
  Link: [https://pan.baidu.com/s/1yMnd-nODR3i9lb7WzHxtCg](https://pan.baidu.com/s/1yMnd-nODR3i9lb7WzHxtCg)  
  Extraction code: **XHCP**  

  **Note:** This link contains not only the dataset, but also the necessary pre-trained model files and required code packages for running the project. Ensure that you download all components to properly set up the environment.

---

## 3. Extracting and Compressing Deep Features

To extract and compress deep features, run the following script:
```bash
python deep_feature_extract_compress.py
```

---

## 4. Feature Selection and Classification

Once the features are extracted and compressed, run the script below for feature selection and classification:
```bash
python feature_selection_and_classification.py
```

---

## 5. Important Notes

- **Time Consumption:**  
  Due to the large size of the training dataset, extracting and compressing deep features using `deep_feature_extract_compress.py` is time-consuming. As a result, we provide pre-compressed deep features stored in the `compress_features.csv` file.

- **Training with Pre-compressed Features:**  
  Researchers can directly execute the `feature_selection_and_classification.py` script based on the pre-compressed features to perform training and obtain the reported metrics.

---

## 6. Acknowledgements

In this project, we utilized open-source code repositories provided by various institutions and researchers. We would like to express our gratitude to all the authors for their contributions.
