import torch, os
import numpy as np
import pytorch_lightning as pl
import sys
from torchmetrics.functional import auroc
fpath= os.path.dirname(os.path.realpath(__file__))
print (sys.path)
# from ief_models.utils import calc_stats, SWA


class Model(pl.LightningModule): 

    def __init__(self, hparams, **kwargs): 
        super().__init__()
        torch.manual_seed(0)
        # self.save_hyperparameters()
        np.random.seed(0)
        self.hparams.update(hparams)

        self.bs = hparams['bs']
        self.lr = hparams['lr']


        self.mi_a= None
        self.mi_n = None
        self.masked_nll_mean = None
        self.masked_kl_mean = None
    def forward(self,**kwargs):
        raise ValueError('Should be overriden')
    def predict(self,**kwargs):
        raise ValueError('Should be overriden')
    def init_model(self,**kwargs):
        raise ValueError('Should be overriden')

    def training_step(self, batch, batch_idx): 
        if self.hparams['anneal'] != -1.: 
            anneal = min(1, self.current_epoch/(self.hparams['max_epochs']*0.5))
            self.hparams['anneal'] = anneal
        else: 
            anneal = 1.
        _, loss  = self.forward(*batch, anneal = anneal) 
        return {'loss': loss}

    def training_epoch_end(self, outputs):  
        reg_losses       = [x['loss'] for x in outputs]
        avg_loss         = torch.stack(reg_losses).mean()
        self.log('tr_loss', avg_loss, prog_bar=True, logger=True)


    def validation_step(self, batch, batch_idx):

        (nelbo, nll, kl, _, _), _ = self.forward(*batch, anneal = 1.)

        Z_t, chi2, _ = self.predict(*batch)

        # if dataloader_idx == 0:
        self.log('val_loss', nelbo, prog_bar=False)
        self.log('val_nll', nll, prog_bar=True)
        self.log('val_kl',kl, prog_bar=False)
        self.log("val_chi2", chi2, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        (nelbo, nll, kl, _, _), _ = self.forward(*batch, anneal = 1.)

        Z_t, chi2, reuslt = self.predict(*batch)

        self.log('test_elbo', nelbo, prog_bar=False, logger=True)
        self.log('test_nll', nll/Z_t.shape[1], prog_bar=True, logger=True)
        self.log('test_kl',kl, prog_bar=False, logger=True)
        self.log("test_chi2", chi2, prog_bar=True, logger=True)
        # print("111111111111")
        self.result =  reuslt

        # self.log("reuslt", reuslt, prog_bar=True, logger=True)
        return (nll/Z_t.shape[1], chi2, reuslt)

    def configure_optimizers(self): 
        if self.hparams['optimizer_name'] == 'adam': 
            opt = torch.optim.Adam(self.parameters(), lr=self.lr) 
            return opt
        elif self.hparams['optimizer_name'] == 'rmsprop': 
            opt = torch.optim.RMSprop(self.parameters(), lr=self.hparams['lr'], momentum=.001)
