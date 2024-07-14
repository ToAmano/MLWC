

import torch

import ml.mlmodel
import importlib
importlib.reload(ml.mlmodel)

# *  モデル（NeuralNetworkクラス）のインスタンス化
model = ml.mlmodel.NET_withoutBN("model_ch", 288, 20, 6)


# cpt = torch.load("model_ch_weight.pth")
# stdict_m = cpt['model_state_dict']
# stdict_o = cpt['opt_state_dict']
# stdict_s = cpt['scheduler_state_dict']

model.load_state_dict(torch.load("model_ch_weight.pth"))

# torchscript
script_model = torch.jit.script(model)

# save model
script_model.save("model_ch_torchscript.pt")
