import argparse
import pandas 
import cupy as cp
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Impute the data using KNN, file must contain only float or int values') 
parser.add_argument('-file',type=str,required=True,dest="file", help="Name of the file to be imputed")
parser.add_argument('-n_neigh',type=str,required=False,dest="n_neigh", help="Number of neighbors in Knn")

args = parser.parse_args()



def replace_value(current_array, array,col,k_neigh=5):
    distancesid_to_curr = cp.argsort(cp.sqrt(cp.nansum((current_array - array) ** 2, axis=1)))
    return cp.nanmean(array[distancesid_to_curr[:k_neigh+1]][:,col]).item() # +1 because distance to himself is 0             

def knnimputer(df, k_neigh=5):
    cp_arr = cp.asarray(df.to_numpy(), dtype=cp.float32)
    for j in tqdm(range(cp_arr.shape[0])):
    for k in range(cp_arr.shape[1]):
        if cp.isnan(cp_arr[j])[k] == True:
            cp_arr[j][k]= replace_value(cp_arr[j],cp_arr,k,k_neigh=k_neigh)
    return cp_arr
            
df = pd.read_csv(str(args.file))
df.dropna(axis=0, how='all')
if args.n_neigh:
    df1 = knnimputer(df,args.n_neigh)
else:
    df1 = knnimputer(df)

df1.savetxt("result.csv", delimiter=";")