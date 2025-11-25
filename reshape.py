import numpy as np
params = dict(np.load("/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-09-24_3412x2500_combin2_2_test1_start_58/2025-08-06_16-09-24_3412x2500_combin2_2/params_0.npz"))  

new_params={}
for k in params.keys():
    if len(params[k].shape)==3:
         new_params[k]=params[k][:60]
    else:
         new_params[k]=params[k]

np.savez("temp.npz", **new_params)