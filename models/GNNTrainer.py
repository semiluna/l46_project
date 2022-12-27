import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.GAT import GATNet

class GNNTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = GATNet(args)

        self.loss_func = nn.CrossEntropyLoss()
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
       
    def forward(self, data, mode='train'):

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"
        
        pred = self.model(data)

        loss = self.loss_func(pred[mask], data.y[mask])
        acc = (pred[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        self.best = {'val_acc': 0, 'epoch': 0}
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode='train')
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        if acc > self.best['val_acc']:
            self.best['val_acc'] = acc
        
        print("Val:[{:.2f}] | Best Val: {:.2f} | Graph Sparsity: {:.2f} | Model sparsitty: {:.2f}"
               .format(acc * 100, 
                        self.best['val_acc'] * 100, 
                        self.graph_spar,
                        self.model_spar))

        self.log("val_acc", acc)
    
    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)