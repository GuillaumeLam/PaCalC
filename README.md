# Participant Calibration Curve (PaCalC)

Repo for 'insert_paper_title'

## Dataset Generation

Place 'data.mat' in folder. GET FROM: [link](https://springernature.figshare.com/collections/A_database_of_human_gait_performance_on_irregular_and_uneven_surfaces_collected_by_wearable_sensors/4892463), the dataset from [A database of human gait performance on irregular and uneven surfaces collected by wearable sensors](https://www.nature.com/articles/s41597-020-0563-y)

Run (in folder `dataset`) to generate Gait on Irregular Surface(GoIS) dataset: 
```
matlab src/mat_to_py_cmptbl_mat.m
python src/py-cmptbl_mat_to_npy.py
```

Run to generate normalized gait on irregular surface (nGoIS): 
```
python src/normalized_dataset.py
```

## Running Code

To run the code:

```
python main.py -v [version]
```

There are three version:
- fast -> single participant (used to make sure everything is running smoothly)
- medium -> 2 dataset folds (used to make sure cross-validation code is running smoothly)
- paper -> full dataset folds with graphs used in the paper

## Re-use of Code

To reuse the code for your personal usage, methods of `util_functions.py` are written to be dataset agnostic. The methods of `main.py` show the usage of the methods for the irregular walking surfaces dataset.

The method `keras_base_model` will need modification based on your model type.

Additionally, `load_data.py` has a caching mechanism of the datasets for seeds ie. datasets split with a seed will be stored in a file for quicker retrieval later on. 

## Sources
- [Generalizability of deep learning models for predicting outdoor irregular walking surfaces](https://doi.org/10.1016/j.jbiomech.2022.111159)
