# Participant Calibration Curve (PaCalC)

Repo for 'insert_paper_title'

## Dataset

Run code from repo [here](https://github.com/Vaibhavshahvr7/Surface-classification-Final) to generate the following file. Place the processed dataset from [here](https://drive.google.com/drive/folders/1XiyOS47Vvt_JM0cCqc-efDANtExbP9mG?usp=share_link) in the folder `PaCalC/dataset`.

## Running Code

To run the code:

```
python main.py -v [version]
```

There are three version:
- fast -> single participant (used to make sure everything is running smoothly)
- medium -> 2 dataset folds (used to make sure cross-validation code is running smoothly)
- paper -> full dataset folds with graphs used in the paper

The main function to utilize is `PaCalC_F1_cv`. This function allows to run cross-validation on the irregular walking surfaces dataset for a generalized calibration curve per label. Onwards, functions are wrappers for the argument parsing & different versions.

## Re-use of Code

To reuse the code for your personal usage, methods of `util_functions.py` are written to be dataset agnostic. The methods of `main.py` show the usage of the methods for the irregular walking surfaces dataset.

The method `keras_base_model` will need modification based on your model type.

Additionally, `load_data.py` has a caching mechanism of the datasets for seeds ie. datasets split with a seed will be stored in a file for quicker retrieval later on. 

## Sources
- [Generalizability of deep learning models for predicting outdoor irregular walking surfaces](https://doi.org/10.1016/j.jbiomech.2022.111159)
