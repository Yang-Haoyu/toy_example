import sys
import os
import argparse
from data.return_dataset import return_data, return_data2, generate_dataset_setting
cwd = os.getcwd()
from torch_attn.model_torch_rec import train_attn_model
from data.syn_dataloader import  SynDataModule2
# from dpm_model.dpm_gene import DPM_vae
# from dpm_model.dpm_gene2 import DPM_vae2
from dpm_model.dpm_gene3 import DPM_vae3
from pytorch_lightning import Callback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from hyperpara import dict_args
from time import time
from baseline_models.ssm import SSM
from baseline_models.dhmm.dhmm import DHMM
import torch
from scipy.stats import chi2_contingency
# from lstm.lstm import lstm
import numpy as np
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []
        self.train_loss = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics['val_loss'].item())
        if 'loss' in trainer.callback_metrics:
            self.train_loss.append(trainer.callback_metrics['loss'].item())

def function_with_args_and_default_kwargs(kwargs, optional_args=None):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args(optional_args)
    return args

# for ttype in ["ief", "ief_g", "gene_attn", "gene_vae", "gene_hybrid"]:
# from scipy.stats.contingency import odds_ratio 
# datasetid = 0
# gpu = 0
# seed = 1
# random_split_seed = 0
# torch.manual_seed(datasetid)



# # callbacks = [metrics_callback, EarlyStopping(monitor='val_nll')]
# # callbacks = [ModelCheckpoint(monitor='val_nll', mode="min"), EarlyStopping(monitor="val_loss", mode="min", patience=3)]
# # callbacks = []
# para_dic = generate_dataset_setting()
# dict_args.update(para_dic[datasetid])
# # dat_syn = return_data(para_dic[datasetid], seed = seed)
# dat_syn = return_data2(para_dic[datasetid], seed = seed)
# dm = SynDataModule2(dict_args, dat_syn, seed = random_split_seed)
# metrics_callback = MetricsCallback()
def run_exp(datasetid, seed, random_split_seed):
    #     for seed in range(10):
    torch.manual_seed(datasetid)
    gpu = None



    # callbacks = [metrics_callback, EarlyStopping(monitor='val_nll')]
    # callbacks = [ModelCheckpoint(monitor='val_nll', mode="min"), EarlyStopping(monitor="val_loss", mode="min", patience=3)]
    # callbacks = []
    para_dic = generate_dataset_setting()
    dict_args.update(para_dic[datasetid])
    # dat_syn = return_data(para_dic[datasetid], seed = seed)
    dat_syn = return_data2(para_dic[datasetid], seed = seed)
    #     for random_split_seed in range(10):
    dm = SynDataModule2(dict_args, dat_syn, seed = random_split_seed)
    result_chi2 = {}
    result_nll = {}

    tic = time()
    print("=============================== exp {} ============================".format(datasetid))
    # for seed, _add in [(1, True), (4, False)]:
    # for _seed, _add, _beta in [(1, True,1000)]:
    # for _seed, _add, _beta in [(1, True,5), (1, True,50), (1, True,100)]:
    _seed = 1
    for _beta in [0.01]:

        for ttype, _add in [ ("pdpm_vae", True), ("ief_g", True)]:
        # for ttype in [ "ief_g"]:
        # for ttype in ["pdpm_attn", "ief_g"]:

            dict_args.update(dm.hparams)
            dict_args.update({ 'ttype': ttype, 'etype': 'lin', "gpu": gpu, "setting": _seed, "add_vae_loss": _add})
            dict_args["_beta"] = _beta
            device = torch.device('cpu')
            # dict_args["setting"] = 2
            model = DPM_vae3(dict_args, **dict_args).to(device)

            model.init_model()


            args = function_with_args_and_default_kwargs(dict_args)

            tb_logger = CSVLogger('lightning_logs', name=ttype + "_" + str(datasetid) + str(seed) + str(random_split_seed))
            callbacks = [EarlyStopping(monitor='val_nll', mode="min")]
            trainer = Trainer.from_argparse_args(args,
                                                deterministic=True,
                                                logger=tb_logger,
                                                callbacks=callbacks,
                                                progress_bar_refresh_rate=1
                                                )
            print("model {}".format(ttype))
            trainer.fit(model, dm)
            scores = trainer.test(model, dataloaders = dm.test_loader)[0]
            result_chi2.setdefault("{} {} {}".format(ttype, _seed, _beta), []).append(scores["test_chi2"])
            result_nll.setdefault("{} {} {}".format(ttype, _seed, _beta), []).append(scores["test_nll"])




    print("=============================== exp {} ============================".format(datasetid))
    dm.setup("train")




    X, Trt, Z_gt = dm.data_tr[:][0], dm.data_tr[:][1], dm.data_tr[:][5]
    x_tr = torch.cat([X, Trt], dim = -1)
    x_tr = list(x_tr.numpy())


    X, Trt, Z_gt = dm.data_val[:][0], dm.data_val[:][1], dm.data_val[:][5]
    x_val = torch.cat([X, Trt], dim = -1)
    x_val = list(x_val.numpy())

    Tmax = np.max([i.shape[0] for i in x_tr])

    trainer = train_attn_model(Tmax, x_tr[0].shape[-1], num_states = para_dic[datasetid]['n_clusters'],
                            num_epochs = 20, batch_size = 100, learning_rate = 5e-3)

    trainer.fit(x_tr, x_val)
    # trainer.predict(x_tr, list(Z_gt))

    X, Trt,  Z_gt = dm.data_test[:][0], dm.data_test[:][1],  dm.data_test[:][5]

    x_tst = torch.cat([X, Trt], dim = -1)

    x_tst = list(x_tst.numpy())

    _nll, chi2, result, chi2_rand = trainer.predict(x_tst, list(Z_gt))
    _nll = _nll/Z_gt.shape[1]
    print("attn result, nll {}, chi2 {}".format(_nll, chi2))

    result_chi2.setdefault("ASSM", []).append(chi2)
    result_nll.setdefault("ASSM", []).append(_nll)





    print("=============================== exp {} ============================".format(datasetid))
    ttype = "DMM"
    dict_args.update({'ttype': ttype, 'etype': 'lin', "gpu": gpu})

    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dict_args.update(dm.hparams)
    model = DHMM(dict_args, **dict_args).to(device)
    model.init_model()



    # dm.setup("train")
    # tr_dat = dm.train_dataloader()
    # dict_args['checkpoint_callback'] = False
    args = function_with_args_and_default_kwargs(dict_args)
    tb_logger = CSVLogger('lightning_logs', name=ttype + "_" + str(datasetid) + str(seed) + str(random_split_seed))
    # args.max_epochs = 1
    callbacks = [EarlyStopping(monitor='val_nll', mode="min")]
    trainer = Trainer.from_argparse_args(args,
                                            deterministic=True,
                                            logger=tb_logger,
                                            callbacks=callbacks,
                                            progress_bar_refresh_rate=1
                                            )
    print("model {}".format(ttype))
    trainer.fit(model, dm)
    scores = trainer.test(model, dataloaders = dm.test_loader)[0]

    result_chi2.setdefault("DHMM", []).append(scores["test_chi2"])
    result_nll.setdefault("DHMM", []).append(scores["test_nll"])





    for ttype in ["ief", "ief_g"]:

        print("=============================== exp {} ============================".format(datasetid))
        print("model {}".format(ttype))



        if gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        dict_args.update(dm.hparams)
        dict_args.update({'ttype': ttype, 'etype': 'lin', "gpu": gpu})
        model = SSM(dict_args, **dict_args).to(device)

        model.init_model()

        args = function_with_args_and_default_kwargs(dict_args)
        tb_logger = CSVLogger('lightning_logs', name=ttype + "_" + str(datasetid) + str(seed) + str(random_split_seed))
        callbacks = [EarlyStopping(monitor='val_nll', mode="min")]
        trainer = Trainer.from_argparse_args(args,
                                                deterministic=True,
                                                logger=tb_logger,
                                                callbacks=callbacks,
                                                progress_bar_refresh_rate=1
                                                )
        print("model {}".format(model.hparams["ttype"]))
        trainer.fit(model, dm)
        scores = trainer.test(model, dataloaders = dm.test_loader)[0]

        result_chi2.setdefault(ttype, []).append(scores["test_chi2"])
        result_nll.setdefault(ttype, []).append(scores["test_nll"])

    toc = time()
    print("time per model: {:.2f} min".format((toc - tic) / 60))

    torch.save(result_chi2, "result/result_chi2_{}_{}_{}".format(datasetid, seed, random_split_seed))
    torch.save(result_nll, "result/result_nll_{}_{}_{}".format(datasetid, seed, random_split_seed))


# run_exp(0, dat_seed, cv_seed)
if __name__ == "__main__" :
    p = argparse.ArgumentParser()
    p.add_argument("--dat_seed")
    p.add_argument("--cv_seed")
    p.add_argument("--did")
    args = p.parse_args()
    dat_seed = int(args.dat_seed)
    cv_seed = int(args.cv_seed)
    did = int(args.did)

    run_exp(did, dat_seed, cv_seed)
#     print(dat_seed)
#     print(cv_seed)
#     run_exp(0, 0, 0)
# parser = argparse.ArgumentParser()  
# parser.add_argument('--did')
# parser.add_argument('--gpu')
# args = parser.parse_args()
# # ttype = 

# run_exp(int(args.did), int(args.gpu))
