Place 'data.mat' in folder. GET FROM: [link](https://springernature.figshare.com/collections/A_database_of_human_gait_performance_on_irregular_and_uneven_surfaces_collected_by_wearable_sensors/4892463), the dataset from [A database of human gait performance on irregular and uneven surfaces collected by wearable sensors](https://www.nature.com/articles/s41597-020-0563-y)

Run to generate Gait on Irregular Surface(GoIS) dataset: 
```
matlab src/mat_to_py_cmptbl_mat.m
python src/py-cmptbl_mat_to_npy.py
```

Run to generate normalized gait on irregular surface (nGoIS): 
```
python src/normalized_dataset.py
```
