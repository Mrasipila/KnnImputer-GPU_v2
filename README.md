# KnnImputer-GPU_v2

40% faster than the version on CPU by sklearn for a dataset of 4 million cells.

Simple tool That implement KNN Imputation for GPUs

## Dependencies

- Cupy
- Pandas
- argparse

## How it works

```
python KnnImputer.py -h
usage: KnnImputer.py [-h] -file FILE [-n_neigh N_NEIGH]

KNN Imputer for GPU

optional arguments:
  -h, --help        show this help message and exit
  -file FILE        Name of the file to be imputed
  -n_neigh N_NEIGH  Number of neighbors in Knn

```

## Example running 

``` 
python KnnImputer.py -file "dataframe.csv" -n_neigh 10
```
