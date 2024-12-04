Lung Prediction Project

Table of Contents
Introduction
Features
Installation
Clone the Repository
Conda Environment Setup
Dataset
Usage
Extract and Compress Deep Features
Feature Selection and Classification
Important Notes
Contributing
Acknowledgements
License
Introduction
Welcome to the Lung Prediction Project repository! This project aims to predict lung conditions using deep learning and machine learning techniques. It leverages the CXR-RANet dataset for training and evaluation.

Features
Deep Feature Extraction: Extracts deep features from chest X-ray images.
Feature Compression: Compresses the extracted features for efficient processing.
Feature Selection: Selects the most relevant features for classification.
Classification: Implements various classification algorithms to predict lung conditions.
Preprocessed Data: Provides pre-extracted and compressed features to save time.
Installation
Clone the Repository
First, clone this repository to your local machine:

bash
Copy code
git clone https://github.com/your-username/your-repository.git
cd your-repository
Conda Environment Setup
We recommend using Conda to manage the project dependencies. Follow the steps below to set up your environment:

Create a Conda Environment:

bash
Copy code
conda create --name lung_predict python=3.8
Activate the Conda Environment:

bash
Copy code
conda activate lung_predict
Install Dependencies and Required Packages:

Ensure you have a requirements.txt file in the root directory of the repository. Install the dependencies using:

bash
Copy code
pip install -r requirements.txt
Dataset
CXR-RANet
The project uses the CXR-RANet dataset, which can be downloaded from Baidu Netdisk. Follow the link below to access the dataset:

Download Link: CXR-RANet on Baidu Netdisk
Extraction Code: XHCP
Note: This dataset is shared by a Baidu Netdisk Super Member (v4).

After downloading, extract the dataset and place it in the designated data directory as specified in the project configuration.

Usage
Extract and Compress Deep Features
Due to the large size of the training dataset, extracting and compressing deep features can be time-consuming. To perform this step, run the following command:

bash
Copy code
python deep_feature_extract_compress.py
This script will process the images and generate compressed deep features stored in compress_features.csv.

Feature Selection and Classification
Once you have the compressed features, you can proceed with feature selection and classification by running:

bash
Copy code
python feature_selection_and_classification.py
This will train the classification models and output the performance metrics.

Important Notes
Time-Consuming Process: Extracting and compressing deep features using deep_feature_extract_compress.py is highly time-consuming due to the large dataset size. To expedite the process, we provide pre-extracted and compressed deep features in the compress_features.csv file.

Using Preprocessed Data: If you prefer to skip the feature extraction step, you can directly use the provided compress_features.csv by running the feature_selection_and_classification.py script. This allows you to train the models and obtain the reported metrics without waiting for feature extraction.

Environment Consistency: Ensure that you are using the same Python version (3.8) and have all the required packages installed to avoid compatibility issues.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

Fork the repository.
Create your feature branch: git checkout -b feature/YourFeature
Commit your changes: git commit -m 'Add some feature'
Push to the branch: git push origin feature/YourFeature
Open a pull request.
Acknowledgements
We utilized open-source code repositories provided by various institutions and researchers. A special thanks to all the authors for their valuable contributions.
CXR-RANet Dataset: Thanks to the contributors of the CXR-RANet dataset for providing the necessary data for this project.
