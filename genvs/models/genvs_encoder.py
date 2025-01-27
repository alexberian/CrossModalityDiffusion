# written by: Alex Berian <berian@arizona.edu>
# our implementation of GeNVS's encoder.

from typing import Optional
from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
import torch

class GeNVSEncoder(DeepLabV3Plus):
    '''
    Modified the DeepLabV3+ Segmentation Model.

    The upsampling in the decoder is replaced with learned convolutional layers
    and skip connections from the encoder layers. 

    The segmentation head is replaced with a feature volume head that takes the
    decoder output and upsamples/reshapes it into a feature volume. This 
    upsampling is also done with learned convolutional layers and skip connections.
    '''
    def __init__(
        self,
        encoder_backbone: str = 'resnet34',
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = 'imagenet',
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        upsample_conv_kernel: int = 3,
        upsample_stage_conv: int = 2,
        volume_features: int = 16,
        volume_depth: int = 64,
        in_channels: int = 3,
        aux_params: Optional[dict] = None,
        disable_batchnorm_and_dropout: bool = True,
    ):
        super().__init__()
        self.latent_size = volume_features

        if encoder_output_stride not in [8, 16]:
            raise ValueError('Encoder output stride should be 8 or 16, got {}'.format(encoder_output_stride))

        # make unmodified encoder
        self.encoder = get_encoder(
            encoder_backbone,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        # make the modified decoder
        self.decoder = ModifiedDLV3PDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            upsample_conv_kernel=upsample_conv_kernel,
            upsample_stage_conv=upsample_stage_conv,
        )

        # make the volume head
        self.volume_head = FeatureVolumeHead(
            decoder_channels=decoder_channels,
            n_features=volume_features,
            depth=volume_depth,
            upsample_conv_kernel=upsample_conv_kernel,
            upsample_stage_conv=upsample_stage_conv,
            encoder_in_channels=in_channels,
        )

        # just leave aux params as None for now
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                **aux_params
            )
        else:
            self.classification_head = None 
        
        # 'We disable batchnorm and dropout THROUGHOUT the feature volume encoder'
        if disable_batchnorm_and_dropout:
            self.apply(self._deactivate_batchnorm_and_dropout)

    def forward(self, x):
        '''
        Modified to use the modified feature volume head instead of
        the segmentation head.
        '''
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        # masks = self.segmentation_head(decoder_output)
        feature_volume = self.volume_head(decoder_output,features)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return feature_volume, labels

        self.latent = feature_volume
        return feature_volume
        
    @staticmethod
    def identity_forward(x):
        return x
            
    @staticmethod
    def _deactivate_batchnorm_and_dropout(m):
        '''
        Disables batchnorm and dropout throughout a model.
        '''
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.Dropout):
            m.eval()
            m.forward = GeNVSEncoder.identity_forward

#----------------------------------------------------------------------------
# GeNVS DeepLabV3++ Encoder Model.

class ModifiedDLV3PDecoderUp(torch.nn.Module):
    '''
    Replaces self.up in the DeepLabV3PlusDecoder class.
    Contains learned convolutional layers and skip connections
    from the encoder layers.

    The old up function outputs a tensor with shape (batch_size, 256, 32, 32)
    '''
    def __init__(
        self,
        out_channels: int = 256,
        kernel_size: int = 3,
        upsample_stage_conv: int = 2
    ):
        super().__init__()

        self.oc = out_channels
        self.ks = kernel_size
        assert(upsample_stage_conv > 0), 'upsample_stage_conv must be greater than 0'
        self.upsample_stage_conv = upsample_stage_conv
        self.make_layers()
    
    def make_layers(self,):
        self.stage1 = self.build_stage(1024,256)
        self.upconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256,256,kernel_size=2,stride=2,padding=0),
            torch.nn.ReLU(),
        )
        self.stage2 = self.build_stage(384,256)
        self.upconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256,256,kernel_size=2,stride=2,padding=0),
            torch.nn.ReLU(),
        )
        self.stage3 = self.build_stage(320,self.oc)

    def build_stage(self,ic,oc):
        '''
        Builds a stage of the convolutional layers between 
        each upsampling ConvTranspose2d.
        '''
        stage = []
        stage.append(torch.nn.Conv2d(ic,oc,kernel_size=self.ks,stride=1,padding='same'))
        stage.append(torch.nn.ReLU())
        for i in range(self.upsample_stage_conv):
            stage.append(torch.nn.Conv2d(oc,oc,kernel_size=self.ks,stride=1,padding='same'))
            stage.append(torch.nn.ReLU())
        return torch.nn.Sequential(*stage)

    def forward(self,aspp_features,features):
        '''
        Features is a list of outputs from each stage in the resnet encoder.
        '''
        # concatenate features[4,5] and aspp_features along the channel dimension
        x = torch.cat([features[4],features[5],aspp_features],dim=1) # (batch_size, 1024, 8, 8)

        # 2x2 conv
        x = self.stage1(x) # (batch_size, 256, 8, 8)
        
        # upconv by 2x with 256 output channels
        x = self.upconv1(x) # (batch_size, 256, 16, 16)

        # concatenate features[3] and x along the channel dimension
        x = torch.cat([features[3],x],dim=1) # (batch_size, 384, 16, 16)

        # 3x3 conv with 256 output channels
        x = self.stage2(x) # (batch_size, 256, 16, 16)

        # upconv by 2x with 256 output channels
        x = self.upconv2(x) # (batch_size, 256, 32, 32)

        # concatenate features[2] and x along the channel dimension
        x = torch.cat([features[2],x],dim=1) # (batch_size, 320, 32, 32)

        # 3x3 conv with 256 output channels
        x = self.stage3(x) # (batch_size, 256, 32, 32)

        return x

class ModifiedDLV3PDecoder(DeepLabV3PlusDecoder):
    '''
    Modified to replace self.up with a ModifiedDLV3PDecoderUp.
    The modified upsampler replaces bilinear upsampling with
    learned convolutional layers and skip connections from
    encoder layers.
    '''
    def __init__(
        self,
        upsample_conv_kernel: int = 3,
        upsample_stage_conv: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.modified_up = ModifiedDLV3PDecoderUp(
            out_channels        = self.out_channels,
            kernel_size         = upsample_conv_kernel,
            upsample_stage_conv = upsample_stage_conv,
        )

    def forward(self, *features):
        '''
        Features is a list of outputs from each stage in the resnet encoder.
            len(features): 6
            [0] torch.Size([69, 3, 128, 128])
            [1] torch.Size([69, 64, 64, 64])
            [2] torch.Size([69, 64, 32, 32])
            [3] torch.Size([69, 128, 16, 16])
            [4] torch.Size([69, 256, 8, 8])
            [5] torch.Size([69, 512, 8, 8])
        '''
        aspp_features = self.aspp(features[-1]) # (batch_size, 256, 8, 8)

        # replace self.up with modified upsampler
        # aspp_features = self.up(aspp_features)
        aspp_features = self.modified_up(aspp_features,features)

        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features) # (batch_size, 256, 32, 32)

        return fused_features

class FeatureVolumeUp(ModifiedDLV3PDecoderUp):
    '''
    Replaces upsampling that is normally in the segmentation head.
    Contains learned convolutional layers and skip connections
    from the encoder layers.
    '''
    def __init__(
        self,
        encoder_in_channels: int = 3,
        **kwargs
    ):
        self.encoder_in_channels = encoder_in_channels
        super().__init__(**kwargs)

    def make_layers(self,):
        self.upconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.oc,self.oc,kernel_size=2,stride=2,padding=0),
            torch.nn.ReLU(),
        )
        self.stage1 = self.build_stage(self.oc+64,self.oc)
        self.upconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.oc,self.oc,kernel_size=2,stride=2,padding=0),
            torch.nn.ReLU(),
        )
        self.stage2 = self.build_stage(self.oc+self.encoder_in_channels,self.oc)

    def forward(self,decoder_output,features):
        # upconv by 2x with 256 output channels
        x = self.upconv1(decoder_output) # (batch_size, 256, 64, 64)

        # concatenate features[1] and x along the channel dimension
        x = torch.cat([features[1],x],dim=1) # (batch_size, 320, 64, 64)

        # 3x3 conv with 320 output channels
        x = self.stage1(x) # (batch_size, 256, 64, 64)

        # upconv by 2x with 320 output channels
        x = self.upconv2(x) # (batch_size, 320, 128, 128)

        # concatenate features[0] and x along the channel dimension
        x = torch.cat([features[0],x],dim=1) # (batch_size, 323, 128, 128)

        # 3x3 conv
        x = self.stage2(x) # (batch_size, 256, 128, 128)

        return x

class FeatureVolumeHead(torch.nn.Module):
    '''
    Takes the decoder output to produce a feature volume.
    Uses 1x1 convolutions to reshape the upsampled decoder output
    to the desired shape, then reshapes the tensor into a volume.
    '''
    def __init__(
        self,
        decoder_channels: int = 256,
        n_features: int = 16,
        depth: int = 64,
        upsample_conv_kernel: int = 3,
        upsample_stage_conv: int = 2,
        encoder_in_channels: int = 3,
    ):
        super().__init__()

        self.n_features = n_features
        self.depth = depth

        self.up = FeatureVolumeUp(
            encoder_in_channels=encoder_in_channels,
            out_channels=decoder_channels,
            kernel_size=upsample_conv_kernel,
            upsample_stage_conv=upsample_stage_conv,
        )
        self.reshape_conv = torch.nn.Conv2d( decoder_channels,
                                       n_features*depth,
                                       kernel_size=1,
                                       stride=1,
                                       padding='same'
                                    )
        self.activation = torch.nn.ReLU()

    def forward(self,decoder_output,features):
        x = self.up(decoder_output,features)
        x = self.reshape_conv(x)
        x = self.activation(x)
        x = torch.reshape(x,(x.shape[0],self.n_features,self.depth,128,128))
        return x
