# Participant Calibration Curve (PaCalC)

Repo for 'Estimating individual minimum calibration for deep-learning with predictive performance recovery: an example case of gait surface classification from wearable sensor gait data'

The PaCalC toolbox is a set of functions to help users calculate calibration curves of individuals for various collected datasets.

## Dataset

Place the processed dataset from [here](https://drive.google.com/drive/folders/1XiyOS47Vvt_JM0cCqc-efDANtExbP9mG?usp=share_link) in the folder `PaCalC/dataset/`. 
(To generate the processed dataset, run code from repo [here](https://github.com/Vaibhavshahvr7/Surface-classification-Final). The repo, Surface-classification-Final, is set to private. Access can be granted by contacting the user Vaibhavshahvr7.)

## Install

Python dependencies are located in `requirements.txt`. They can be installed with `pip install -r requirements.txt`. To get matplotlib plotting in linux, you might need to install additional libraries w/ `sudo apt-get install tcl-dev tk-dev python-tk python3-tk` (for Debian systems).

## Running Code

To run the code:

```
python main.py -v [version] -m [model_type]
```

There are multiple version:
- demo -> generate graphs out of the box, with no additional downloads!
- fast -> single participant (used to make sure everything is running smoothly)
- medium -> 2 dataset folds (used to make sure cross-validation code is running smoothly)
- paper -> full dataset folds with graphs used in the paper

There are two model architechtures:
- ANN -> feed-forward neural network 
- CNN -> convolutional neural network

The ANN takes ~1k seconds (or ~16 minutes) and CNN takes ~8k seconds (or ~2.2 hours) to generate the paper figures.

The main function to utilize is `PaCalC_F1_cv`. This function allows to run cross-validation on the irregular walking surfaces dataset for a generalized calibration curve per label. Onwards, functions are wrappers for the argument parsing & different versions.

## Expected Output

The two main outputs of this toolbox are two graphs: 1. A calibration curve averaged over all surfaces. 2. A calibration curve for each surface (9). For both of these figures, the calibration curves mean and standard variance (blue) are displayed and reference points are added: the performance of a model trained with subject-wise split (green) and an other model trained with random-wise split (red). Example graphs are shown here:

![averaged-surfaces](readme_fig/PaCalC(dtst_cv%3D2).png)
![all-surfaces](readme_fig/PaCalC_all-surfaces(dtst_cv%3D2).png)

Additionally, plotting of these two graphs is performed per participant for potentially further investigation. 

## Re-use of Code

To reuse the code for your personal usage, methods of `util_functions.py` are written to be dataset agnostic, this file/these methods can be copied over to any project. The methods of `main.py` show the usage of the PaCalC methods for the irregular walking surfaces dataset.

The method `keras_base_model` will need modification based on your model type.

Additionally, `load_data.py` has a caching mechanism of the datasets for seeds ie. datasets split with a seed will be stored in a file for quicker retrieval later on. 

## Running unit tests

To run the unit tests, run the following command:
`python test.py`

## Sources
- [Generalizability of deep learning models for predicting outdoor irregular walking surfaces](https://doi.org/10.1016/j.jbiomech.2022.111159)
