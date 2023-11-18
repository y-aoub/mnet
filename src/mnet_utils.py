from torch import nn
from mnet_layers import MNetDown, MNetConvRelu, MNetConcat, MNetMaxpool, MNetDeconv, MNetConvSig, MNetUp
import json


class MNetUtils:
    def __init__(self, file_path):
        self.file_path = file_path
        self.created_layers = []
        
    def read_json(self):
        data = []
        for path in self.file_path:
            with open(path, 'r') as json_file:
                loaded_json = json.load(json_file)
            data.append(loaded_json)
        return data
    
    def load_json(self):
        json_data = self.read_json()
        for layers_data in json_data:
            self.created_layers.append(self.create_seq_layers(layers_data))
        return self.created_layers
    
    def create_layer(self, layer_name: str, layer_param: {list, None}):
        if "down" in layer_name:
            return MNetDown(kernel_size = layer_param)
        elif "conv_relu" in layer_name:
            return MNetConvRelu(conv_relu = layer_param)
        elif "concat" in layer_name:
            return MNetConcat(dim = layer_param)
        elif "maxpool" in layer_name:
            return MNetMaxpool(kernel_size = layer_param)
        elif "deconv" in layer_name:
            return MNetDeconv(deconv = layer_param)
        elif "conv_sig" in layer_name:
            return MNetConvSig(conv_sig = layer_param)
        elif "up" in layer_name:
            return MNetUp(scale_factor = layer_param)
        
    def create_layers(self, layers_seq):
        created_layers_seq = dict(map(lambda layer: (layer[0], self.create_layer(layer[0], layer[1])), layers_seq.items()))
        return created_layers_seq
    
    def create_seq_layers(self, layers_seqs):
        created_layers_seqs = list(map(lambda layers_seq: self.create_layers(layers_seq), layers_seqs))
        return created_layers_seqs
    
if __name__ == "__main__":
    json_file_path = ['../encoder.json']
    mnet_utils = MNetUtils(file_path=json_file_path)
    json_data = mnet_utils.read_json()[0]
    
    created_layers = mnet_utils.create_seq_layers(json_data)
    print(created_layers)
    
    