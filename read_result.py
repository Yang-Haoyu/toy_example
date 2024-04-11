import torch
import numpy as np
datasetid =0
for seed in range(6):
    print("=================={}=========================".format(seed))
    for datasetid in range(3):
        print("--------------------------------{}--------------------------------".format(datasetid))

        result_chi2 = {}
        result_nll = {}
        for random_split_seed in range(10):
            try:

                result_chi2_tmp =torch.load("result/result_chi2_{}_{}_{}".format(datasetid, seed, random_split_seed))
                result_nll_tmp = torch.load("result/result_nll_{}_{}_{}".format(datasetid, seed, random_split_seed))
            except:
                continue
            for k,v in result_chi2_tmp.items():
                result_chi2.setdefault(k, []).append(result_chi2_tmp[k][0])
                result_nll.setdefault(k, []).append(result_nll_tmp[k][0])

        for k,v in result_chi2.items():
            print("{}: {:.2f}\pm{:.2f}".format(k, np.mean(v), np.std(v)))

        for k,v in result_nll.items():
            print("{}: {:.2f}\pm{:.2f}".format(k, np.mean(v), np.std(v)))

for datasetid in range(3):
    print("=================={}=========================".format(datasetid))
    result_chi2 = {}
    result_nll = {}
    for seed in range(6):

        for random_split_seed in range(10):
            try:
                result_chi2_tmp =torch.load("result/result_chi2_{}_{}_{}".format(datasetid, seed, random_split_seed))
                result_nll_tmp = torch.load("result/result_nll_{}_{}_{}".format(datasetid, seed, random_split_seed))
            except:
                continue
            for k,v in result_chi2_tmp.items():
                result_chi2.setdefault(k, []).append(result_chi2_tmp[k][0])
                result_nll.setdefault(k, []).append(result_nll_tmp[k][0])

    for k,v in result_chi2.items():
        print("{}: {:.2f}\pm{:.2f}".format(k, np.mean(v), np.std(v)))

    for k,v in result_nll.items():
        print("{}: {:.2f}\pm{:.2f}".format(k, np.mean(v), np.std(v)))
        
     
    
