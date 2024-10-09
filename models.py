import argparse

import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, sigmoid, optim
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy


class Model(pl.LightningModule):

    def __init__(self, args : argparse.Namespace, num_genes, hidden=512):
        super().__init__()

        self.args = args
        self.encoder = timm.create_model(args.timm, pretrained=True, num_classes=0)
        embedding_size = self.encoder.num_features
        print(f'Embedding size from encoder : {embedding_size}')

        self.accuracy = BinaryAccuracy()
        #self.criterion = nn.BCELoss()
        #self.criterion = nn.MSELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.BCEWithLogitsLoss(pos_weight=2*torch.ones((embedding_size,)))
        self.classifier = nn.Sequential(
            #nn.Linear(embedding_size, hidden),
            #nn.ReLU(),
            #nn.Linear(hidden, num_genes),
            nn.Dropout(self.args.dropout),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, num_genes),
        )


    def forward(self, tiles):
        feats = self.encoder(tiles)
        logits = self.classifier(feats)
        return logits


    def training_step(self, batch) -> STEP_OUTPUT:
        tiles, expressions = batch
        logits = self.forward(tiles)
        #preds = sigmoid(logits)
        #loss = self.criterion(preds, expressions)       # for classification
        loss = self.criterion(logits, expressions)       # for classification, with BCEWithLogitsLoss, which does sigmoid internally
        #loss = self.criterion(logits, expressions)       # for regression
        self.log('train_loss', loss)

        #acc = self.accuracy(logits, expressions)
        #acc = self.accuracy(torch.clip(logits,0.0, 1.0), expressions)
        #self.log('train_acc', acc)
        return loss

    def validation_step(self, batch) -> STEP_OUTPUT:
        tiles, expressions = batch
        logits = self.forward(tiles)
        #preds = sigmoid(logits)
        #loss = self.criterion(preds, expressions)
        loss = self.criterion(logits, expressions)
        return loss


    def configure_optimizers(self):
        #opt = optim.SGD(self.model.parameters(), lr=0.1)
        params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        opt = optim.Adam(params, lr=0.00001)
        #opt = optim.SGD(params, lr=0.001)
        return opt


