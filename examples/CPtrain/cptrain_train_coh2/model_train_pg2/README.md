# Train
CPtrain.py train -i input_coh.yaml

# Test
CPtrain.py test -m model_coh/model_coh_torchscript.pt -x traj/test.xyz -i pg2asym.mol -b COH
