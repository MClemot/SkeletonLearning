import torch
import os
from nn import pretrain

os.environ['KMP_DUPLICATE_LIB_OK']='True'

net = pretrain(dim_hidden=64, num_layers=6, skip=[], lr=2e-5, batch_size=25000, epochs=5000, activation="ReLU")
torch.save(net, "Pretrained/pretrained_{}_{}_{}.net".format(64, 6, "ReLU"))

net = pretrain(dim_hidden=64, num_layers=6, skip=[], lr=2e-5, batch_size=25000, epochs=5000, activation="Sine")
torch.save(net, "Pretrained/pretrained_{}_{}_{}.net".format(64, 6, "Sine"))

net = pretrain(dim_hidden=64, num_layers=6, skip=[], lr=2e-5, batch_size=25000, epochs=5000, activation="SoftPlus")
torch.save(net, "Pretrained/pretrained_{}_{}_{}.net".format(64, 6, "SoftPlus"))