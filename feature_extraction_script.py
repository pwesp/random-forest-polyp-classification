import csv
import json
import nrrd
import numpy as np
import os
import pandas as pd
from   radiomics import featureextractor



# File which contains the extraction parameters for Pyradiomics
params_file  = 'source/feature_extraction_parameters_example.yaml'

# File which contains the paths to CTC images and segmentations
image_info_file = 'example_image_info_train.csv'

# File where the extracted pyradiomics features will be stored
output_path  = 'extracted_radiomics_features/extracted_example_features_train.csv'

# Overwrite feature file if it already exists?
overwrite = True

# Remove previously calculated feautures
if os.path.exists(output_path) and overwrite:
        print('Remove {}'.format(output_path))
        os.remove(output_path)

# Collect scan info
df_image_info = pd.read_csv(image_info_file, sep=',')

# Run main and extract pyradiomics features
if __name__ == '__main__':

    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    print(extractor)

    # Initialize header variable for feature vector
    headers = None
    
    # Loop through scan-mask pairs
    for idx, (imageFilepath, maskFilepath) in enumerate(zip(df_image_info['image'], df_image_info['mask'])):
        
        # Get filepaths from dictionary
        print('\nImage file path: {}'.format(imageFilepath))
        print('Mask file path: {}'.format(maskFilepath))
        
        # Sanity checks
        # 1. Do image and mask have same size?
        # 2. Is mask [0,1]?
        imageData, header = nrrd.read(imageFilepath)
        maskData, header = nrrd.read(maskFilepath)
        if not np.array_equal(maskData.shape, imageData.shape):
            print('Warning: mask does not fit image. Go to next scan.')
            continue
        if not np.array_equal(np.unique(maskData), np.array([0,1])):
            print('Warning: mask contains elements other than [0,1]. Go to next scan.')
            continue
            
        print('Image shape {}'.format(imageData.shape))
        print('Mask shape  {}'.format(maskData.shape))
        
        # Extract features
        try:
            print('Extract features...')
            featureVector = extractor.execute(imageFilepath, maskFilepath)

            with open(output_path, 'a') as outputFile:
                writer = csv.writer(outputFile, lineterminator='\n')
                if headers is None:
                    headers = list(featureVector.keys())
                    writer.writerow(headers)

                row = []
                for h in headers:
                    row.append(featureVector.get(h, 'N/A'))
                writer.writerow(row)

        except Exception:
            print('EXCEPTION: Feature extraction failed!')