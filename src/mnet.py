from torch import nn
import torch
from mnet_utils import MNetUtils

ENCODER_PARAMS_PATH = ["./layers/encoder_multi_scale.json", "./layers/encoder_middle.json"]
BOTTLENECK_PARAMS_PATH = ["./layers/bottleneck.json"]
DECODER_PARAMS_PATH = ["./layers/decoder_middle.json", "./layers/decoder_side_out.json"]


class MNetEncoder(nn.Module):
    def __init__(self, encoder_params_path = ENCODER_PARAMS_PATH):
        super(MNetEncoder, self).__init__()
        
        self.encoder_multi_scale_out = []
        self.encoder_conv_out = []
           
        self.mnet_utils = MNetUtils(file_path = encoder_params_path)
        self.encoder_created_layers = self.mnet_utils.load_json()
        
    
    def forward(self, x):
        for i, encoder_created_layers in enumerate(self.encoder_created_layers):
            if i == 0:
                for j, layers_seq in enumerate(encoder_created_layers):
                    z = x
                    for _, layer_name in enumerate(layers_seq.keys()):
                        z = layers_seq[layer_name](z)
                    self.encoder_multi_scale_out.append(z)
            
            elif i == 1:
                for j, layers_seq in enumerate(encoder_created_layers):
                    for l, layer_name in enumerate(layers_seq.keys()):
                        if "concat" in layer_name:
                            x = layers_seq[layer_name](x, y=self.encoder_multi_scale_out[j])
                        else:
                            x = layers_seq[layer_name](x)
                        if l == len(layers_seq.keys()) - 2:
                            self.encoder_conv_out.append(x)
        return x, self.encoder_conv_out
    
class MNetBottleneck(nn.Module):
    def __init__(self, bottleneck_params_path = BOTTLENECK_PARAMS_PATH):
        super(MNetBottleneck, self).__init__()
        
        self.mnet_utils = MNetUtils(file_path = bottleneck_params_path)
        self.bottleneck_created_layers = self.mnet_utils.load_json()
    
    def forward(self, x):
        for _, bottleneck_created_layers in enumerate(self.bottleneck_created_layers):
            for layers_seq in bottleneck_created_layers:
                for layer_name in layers_seq.keys():
                    x = layers_seq[layer_name](x)
        return x
    
class MNetDecoder(nn.Module):
    def __init__(self, encoder_conv_out, decoder_params_path = DECODER_PARAMS_PATH):
        super(MNetDecoder, self).__init__()
        
        self.decoder_conv_out = []
        self.decoder_side_out = []
        
        self.encoder_conv_out = list(reversed(encoder_conv_out))
        
        self.mnet_utils = MNetUtils(file_path = decoder_params_path)
        self.decoder_created_layers = self.mnet_utils.load_json()
        
    def forward(self, x):
        
        for i, decoder_created_layers in enumerate(self.decoder_created_layers):
            
            if i == 0:
                for j, layers_seq in enumerate(decoder_created_layers):
                    for l, layer_name in enumerate(layers_seq.keys()):
                        if "concat" in layer_name:
                            x = layers_seq[layer_name](x, y=self.encoder_conv_out[j])
                        else:
                            x = layers_seq[layer_name](x)
                        if l == len(layers_seq.keys()) - 2:
                            self.decoder_conv_out.append(x)
            
            elif i == 1:
                for j, layers_seq in enumerate(decoder_created_layers):
                    
                    for l, layer_name in enumerate(layers_seq.keys()):
                        z = self.decoder_conv_out[-j-1]
                        z = layers_seq[layer_name](z)
                        print(layer_name, " → ", z.shape)
                    
                    self.decoder_side_out.append(z)
        
        return list(reversed(self.decoder_side_out))


if __name__ == "__main__":
    test_image = torch.torch.randn([1, 3, 400, 400])
    
    print("\nENCODER:")
    mnet_encoder = MNetEncoder()
    final_maxpool_out, encoder_conv_out = mnet_encoder(test_image)
    
    mnet_bottleneck = MNetBottleneck()
    final_deconv_out = mnet_bottleneck(final_maxpool_out)
    print("\nBOTTLENECK:")
    print("output shape: ", final_deconv_out.shape)
    
    mnet_decoder = MNetDecoder(encoder_conv_out=encoder_conv_out)
    decoder_side_out = mnet_decoder(final_deconv_out)
    print("\nDECODER ELEMENTS:")
    for _, elem in enumerate(decoder_side_out):
        print(f"output shape elem_{_}: ", elem.shape)
      
                
                
                    
                    
                        
                    
                    
             


        