https://github.com/Spidercoder/P5-Project

To prepare project:
* Get data files from http://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
* Put data files in data/raw
* Compile features.pyx using Cython
* Create folder "hugin" and add Hugin python library files within
* Run preprocessing/txt_to_csv.py
* Run preprocessing/data_manipulation.py
* Set modified/original type in configuration.py

To generate feature plots to assist feature engineering:
* Run plotting/feature_plotting.py

To generate correlation matrix and feature importance plots:
* Run plotting/feature_selection.py

To generate plots for model selection and results:
* Run run_models.py to generate results
* Run model_selection.py to generate test results
* Run plotting/model_plotting.py