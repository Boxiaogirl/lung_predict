# lung_predict
At first, enter into the code file directory by cd command

Settings for Conda environment: 

Create conda envs:

conda create --name lung_predict python=3.8

activate the conda envs:

conda activate lung_predict

install depndencies and required packages:

pip install -r ./requirements.txt



extact and comnpress deep features, run as follows:

python deep_feature_extract_compress.py

feature selection and classification by run as follows:

python feature_selection_and_classification.py


***********Due to the large size of the training dataset, the process of extracting and compressing deep features using deep_feature-extract_compressed.py is very time-consuming. Therefore, we provide compressed deep features, which are stored in the compress_features.csv file. Based on this file, researchers can directly execute features_selection_and_classification.py for training and get the reported metrics.
***********In the project, we utilized open-source code repositories provided by institutions or researchers, thanks to all authors.

