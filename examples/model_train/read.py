


# read input yaml file
import yaml
with open('test.yaml') as file:
    yml = yaml.safe_load(file)
    print(yml)
    
print(yml["model"])
print(yml["model"]["modelname"])
print(yml["model"]["nfeature"])
print(yml["model"]["M"])
