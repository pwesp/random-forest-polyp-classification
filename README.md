# random-forest-polyp-classification

This repository contains Python scripts and Jupyter notebooks to make (binary) predictions based on radiomics features extracted from CT images using a random forest classifier.

The code has originally been developed for the publication "*Machine Learning-based Differentiation of Benign and Premalignant Colorectal Polyps Detected with CT Colonography in an Asymptomatic Screening Population: A Proof-of-Concept Study*" by Grosu, Wesp, et al. [Radiology, 2021] ([https://doi.org/10.1148/radiol.2021202363](https://doi.org/10.1148/radiol.2021202363)). In this study a random forest model was trained to predict the histopathological class (benign vs. premalignant) of colorectal polyps detected in 3D CT colonography images. However, the code can easily be adjusted to solve other Radiomics tasks and classification problems.

![radiomics](https://user-images.githubusercontent.com/56682642/154975607-7442c01e-d464-4dc8-aa80-5b52178f322f.png)

A description of how to use the code in this repository can be found in section 0. to 3. below. For demonstration and test purposes, artificial training and test data is provided in the folder 'data'. The artificial data consists of 25 randomly generated images and segmentation masks (numpy arrays of size 50x50x50) for training and testing and a set of random labels (0 vs. 1) for each dataset. Radiomics features can be extracted for both datasets and a random forest model can be trained and tested.

The random forest model which has been trained and evaluated in the publication is provided in 'trained_models/random_forest_polyp_classification_model.joblib'. The parameter file to set up the Pyradiomics [1] feature extractor used in the publication can be found at 'source/feature_extraction_parameters_polyp_classification.yaml'.

## 0. Prerequisits

The following things need to be at hand in order to use the code in this repository:

1. CT scans containing **volumes of interests** (VOIs) which should be subject to binary classification  (0 vs. 1)
2. A **segmentation mask** (or multiple segmentation masks) of each VOI
3. Ground truth **labels** (0, 1) for each VOI
4. Images, segmentations and labels are expected to be separated into a training dataset and a test dataset
    - Training images and segmentations are stored in: 'data/training_data'
    - Test images and segmentations are stored in: 'data/test_data'
    - Training labels are stored in: 'data/labels_train.csv'
    - Test labels are stored in: 'data/labels_test.csv'
5. A **Python environment** running the Python version and containing the modules and packages specified in the file 'conda_environment.yaml'

The code is designed for three-dimensional image data (CT scans), but may be adopted for two-dimensional images.

## 1. Pyradiomics feature extraction

First, we extract a set of **Pyradiomics features** from the segmented VOIs in the CT images. This step needs to be performed the training dataset and the test dataset.

0. Images and segmentations are expected to be stored in the "nearly raw raster data" format, i.e. stored as '.nrrd' files
1. Start with the training dataset
2. Create a '.csv' file containing the exact file paths of images and segmentations for each VOI-segmentation pair. Two examples for such a file can be found in 'example_image_info_train.csv' and 'example_image_info_test.csv'
3. Run the Python script 'feature_extraction_script.py' to perform a pyradiomics feature extraction. The '.csv' file from step 2. has to be specified in the variable 'image_info_file'. Also, an output file for the extracted features has to be specified in the variable 'output_path'.
    - The extracted features are expected to be stored in the folder 'extracted_radiomics_features' as 'extracted_example_features_train.csv' or 'extracted_example_features_test.csv', respectively
5. If you just extracted features for the training dataset, go back and repeat steps 1. to 3. for the test dataset.

## 2. Train the random forest model

Second, we train a scikit-learn [2] RandomForestClassifier model.

1. Create a '.csv' file containing the ground truth labels (0, 1) for the training dataset. An example for such a file can be found in 'data/example_labels_train.csv'.
2. Run the 'train_random_forest.ipynb' notebook. Specify the output file from the pyradiomics feature extraction for the training dataset (step 1.3) in the variable 'feature_file_training_set' and the label file (step 2.1) in the variable 'label_file_training_set'.
    - If succesfull, a trained random forest model will be stored under 'trained_models/trained_example_random_forest_model.joblib'.

## 3. Test the random forest model

Third, we test the model.

1. Create a '.csv' file containing the ground truth labels (0, 1) for the test dataset. An example for such a file can be found 'data/example_labels_test.csv'.
2. Run the 'test_random_forest.ipynb' notebook. Specify the output file from the pyradiomics feature extraction for the test dataset (step 1.3) in the variable 'feature_file_test_set' and the label file (step 2.1) in the variable 'label_file_test_set'.

## Resources

1. Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. [https://doi.org/10.1158/0008-5472.CAN-17-0339](https://doi.org/10.1158/0008-5472.CAN-17-0339).
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
