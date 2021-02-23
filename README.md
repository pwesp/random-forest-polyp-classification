# random-forest-polyp-classification

This repository contains Python scripts and Jupyter notebooks to make (binary) predictions based on radiomics features, extracted from CT images, using a random forest classifier. The code has been developed in order to predict the histopathological class (benign vs. premalignant) of colorectal polyps detected in 3D CT colonography images. However, the code may be applied to other Radiomics tasks and classification problems.

A description of how to use the code in this repository can be found in section 1. to 3. below. For demonstration and test purposes, an artificial set of radiomics features and target labels containing random numbers are provided in the folder 'data'.

## 0. Prerequisits

The following things need to be at hand in order to use the code in this repository:

- CT images containing region of interests (ROIs) which should be classified (0 vs. 1)
- A segmentation (or multiple segmentations) of each ROI
- Images and segmentations need to be separated into a training dataset and a test dataset
- A Python environment running the Python version and containing the modules and packages specified in 'environment.yaml'

## 1. Pyradiomics feature extraction

First, we extract Pyradiomics features from the segmented regions of interests in the CT images. This step needs to be performed twice, once for the training dataset and then again for the test dataset.

0. Start with the training dataset.
1. Make sure images and segmentations are stored in the "nearly raw raster data" format, i.e. stored as '.nrrd' files
2. Create a '.csv' file containing the exact paths of images of polyps and their corresponding segmentation(s). An example for such a file can be found in 'data/training_features.csv'
3. Run the Python script 'feature_extraction_script.py' to perform a pyradiomics feature extraction. The '.csv' file from step 2. has to be specified in the variable 'image_info_file'. Also, an output file for the extracted features has to be specified in the variable 'output_path'.
4. If you just extracted features for the training dataset, go back and repeat steps 1. to 3. for the test dataset.

## 2. Train the random forest model

1. Create a '.csv' file containing the ground truth labels of the class you want to predict for the training dataset. An example for such a file can be found 'data/training_labels.csv'.
2. Run the 'train_random_forest.ipynb' notebook. Specify the output file from the pyradiomics feature extraction for the training dataset (step 1.3) in the variable 'feature_file_training_set' and the label file (step 2.1) in the variable 'label_file_training_set'.
3. If succesfull, a trained random forest model will be stored under 'trained_models/trained_random_forest_model.joblib'.

## 3. Test the random forest model

1. Create a '.csv' file containing the ground truth labels of the class you want to predict for the test dataset. An example for such a file can be found 'data/test_labels.csv'.
2. Run the 'test_random_forest.ipynb' notebook. Specify the output file from the pyradiomics feature extraction for the test dataset (step 1.3) in the variable 'feature_file_test_set' and the label file (step 2.1) in the variable 'label_file_test_set'.
