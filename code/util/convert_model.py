import sys
import torch

def convert_model(input_path, out_path, prefix='oan'):
    model = torch.load(input_path+'/model_best.pth')
    new_model = {}
    new_model['epoch'] = model['epoch']
    new_model['state_dict'] = {}
    for key in model['state_dict'].keys():
        new_key = prefix+'.'+key
        new_model['state_dict'][new_key] = model['state_dict'][key]
    torch.save(new_model, out_path+'/model_best.pth')
if __name__ == "__main__":
    convert_model(sys.argv[1], sys.argv[2])
