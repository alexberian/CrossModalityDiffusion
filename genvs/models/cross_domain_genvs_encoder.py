# written by: Alex Berian <berian@arizona.edu>
# our implementation of GeNVS"s encoder.

import sys

from typing import Optional
from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus
from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from .genvs_encoder import GeNVSEncoder
import torch

# encoder imports
from segmentation_models_pytorch.encoders import encoders
import torch.utils.model_zoo as model_zoo
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
from torchvision.models.resnet import conv1x1


class AdditiveEmbedding(torch.nn.Module):
    """
    Applies an additive embedding to a tensor across the channel dimension.
    Assumes the channel dimension is the second dimension of the input tensor.
    """
    def __init__(self, embedding_dim, input_channels, activation_layer = torch.nn.Identity()):
        super().__init__()
        self.emb_match = torch.nn.Linear(embedding_dim, input_channels)
        assert(isinstance(activation_layer, torch.nn.Module)), "Activation layer must be a torch.nn.Module. Got type: {}".format(type(activation_layer))
        self.activation = activation_layer


    def forward(self, tensor, emb):
        """
        Applies the additive embedding to the input tensor.
        """
        # default to return the tensor if no embedding is provided
        if emb is None:
            return tensor

        # make sure shapes are OK
        assert(tensor.shape[1] == self.emb_match.out_features), "Tensor channel dimension (second dim) must match embedding dimension. Expected: {}, Got: {}".format(self.emb_match.out_features, tensor.shape[1])
        assert(tensor.shape[0] == emb.shape[0]), "Batch size of tensor and embedding must match. Got tensor: {}, embedding: {}".format(tensor.shape[0], emb.shape[0])
        assert(emb.shape[1] == self.emb_match.in_features), "Embedding dimension must match the input dimension of the embedding matching layer. Expected: {}, Got: {}".format(self.emb_match.in_features, emb.shape[1])

        # apply the embedding
        x = self.emb_match(emb)
        x = self.activation(x)
        x = x.reshape(tensor.shape[0],tensor.shape[1],1,1)
        return tensor + x



def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, embedding_dim=64, **kwargs):
    """
    Replacement for the get_encoder function from segmentation_models_pytorch.
    Only allows for resnet34 encoder.
    """

    assert name == "resnet34"

    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(embedding_dim=embedding_dim)
    encoder = ModifiedResNetEncoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder



class ModifiedResNetEncoder(ResNetEncoder):
    """
    Modify the ResNetEncoder fron segmentation_models_pytorch to allow for the addition of an embedding.
    """
    def __init__(self, out_channels, depth=5, embedding_dim=64, **kwargs):
        self.embedding_dim = embedding_dim

        super().__init__(out_channels, depth=depth, **kwargs)

        # add embedding matching layers
        self.additive_embedding = AdditiveEmbedding(embedding_dim, 3)


    def forward(self, x, emb = None):
        """
        Modified stage call to allow for the addition of an embedding.
        """
        features = []

        # stage 0
        features.append(x)

        # stage 1
        x = self.additive_embedding(x, emb)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        # stage 2
        x = self.maxpool(x)
        x = self.layer1(x,emb=emb)
        features.append(x)

        # stage 3
        x = self.layer2(x,emb=emb)
        features.append(x)

        # stage 4
        x = self.layer3(x,emb=emb)
        features.append(x)

        # stage 5
        x = self.layer4(x,emb=emb)
        features.append(x)

        return features


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Replaced with a class that allows for an embedding to be passed to the block.
        Addition of embedding happens before each same-dimension in/out conv.
        """
        return ModifiedResNetEncoderLayer(
            self,
            block, planes, blocks, stride, dilate,
        )



class ModifiedResNetEncoderLayer(torch.nn.Module):
    """
    Replacement for the _make_layer function in the ResNetEncoder class.
    Also allows for the addition of an embedding.
    TODO: add embedding
    """
    def __init__( self, parent_encoder,
            block, planes, blocks, stride, dilate, 
        ):
        super().__init__()

        # original code with some compatibility modifications
        norm_layer = parent_encoder._norm_layer
        downsample = None
        previous_dilation = parent_encoder.dilation
        if dilate:
            parent_encoder.dilation *= stride
            stride = 1
        if stride != 1 or parent_encoder.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(parent_encoder.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(parent_encoder.inplanes, planes, stride, downsample, parent_encoder.groups,
                            parent_encoder.base_width, previous_dilation, norm_layer))
        parent_encoder.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(parent_encoder.inplanes, planes, groups=parent_encoder.groups,
                                base_width=parent_encoder.base_width, dilation=parent_encoder.dilation,
                                norm_layer=norm_layer))

        self.layers = torch.nn.ModuleList(layers)

        # embedding related variables
        self.init_embedding_layers(parent_encoder.embedding_dim)


    def init_embedding_layers(self, embedding_dim):
        """
        Initialize the embedding layers.
        """
        # find out input size to each block
        input_dims = []
        for layer in self.layers:
            input_dims.append(layer.conv1.in_channels)

        # create additive embedding layer for each unique input size
        unique_input_dims = list(set(input_dims))
        additive_embedding_layers = []
        for dim in unique_input_dims:
            additive_embedding_layers.append(AdditiveEmbedding(embedding_dim, dim))
        self.additive_embedding_layers = torch.nn.ModuleList(additive_embedding_layers)

        # create a map from input size to additive embedding layer
        self.embed_layer_input_map = dict(zip(unique_input_dims, self.additive_embedding_layers))


    def forward(self, x, emb = None):
        """
        add embedding before each resnet block
        """
        for layer in self.layers:
            x = self.embed_layer_input_map[layer.conv1.in_channels](x, emb)
            x = layer(x)
        return x






class CrossDomainGeNVSEncoder(GeNVSEncoder):
    """
    basically any regular convolution steps will include the domain embeddings
    """
    def __init__(
        self,
        encoder_backbone: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
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
        embedding_dim: int = 64,
    ):
        torch.nn.Module.__init__(self)
        self.latent_size = volume_features

        self.embedding_dim = embedding_dim

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        # make unmodified encoder
        self.encoder = get_encoder(
            encoder_backbone,
            in_channels=in_channels,
            depth=encoder_depth,
            # weights=encoder_weights,
            weights=None,
            output_stride=encoder_output_stride,
            embedding_dim=embedding_dim,
        )

        # make the modified decoder
        self.decoder = ModifiedDLV3PDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            upsample_conv_kernel=upsample_conv_kernel,
            upsample_stage_conv=upsample_stage_conv,
            embedding_dim=embedding_dim,
        )

        # make the volume head
        self.volume_head = FeatureVolumeHead(
            decoder_channels=decoder_channels,
            n_features=volume_features,
            depth=volume_depth,
            upsample_conv_kernel=upsample_conv_kernel,
            upsample_stage_conv=upsample_stage_conv,
            encoder_in_channels=in_channels,
            embedding_dim=embedding_dim,
        )

        # just leave aux params as None for now
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                **aux_params
            )
        else:
            self.classification_head = None 
        
        # "We disable batchnorm and dropout THROUGHOUT the feature volume encoder"
        if disable_batchnorm_and_dropout:
            self.apply(self._deactivate_batchnorm_and_dropout)

    def forward(self, x, emb=None):
        """
        Modified to use the modified feature volume head instead of
        the segmentation head.
        """
        self.check_input_shape(x,emb)

        features = self.encoder(x, emb = emb)
        decoder_output = self.decoder(features, emb = emb)

        # masks = self.segmentation_head(decoder_output)
        feature_volume = self.volume_head(decoder_output,features, emb=emb)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return feature_volume, labels

        self.latent = feature_volume
        return feature_volume

    
    def check_input_shape(self, x, emb):
        """
        Call's superclass's check_input_shape method for x, then checks the shape of emb.
        """
        super().check_input_shape(x)
        if emb is not None:
            if emb.shape[1] != self.embedding_dim:
                raise ValueError("Domain embedding must have shape (batch_size, embedding_dim). Got shape: {}".format(emb.shape))
            if emb.shape[0] != x.shape[0]:
                raise ValueError("Batch size of input tensor and domain embedding must match. Got input tensor: {}, domain embedding: {}".format(x.shape[0], emb.shape[0]))
            if len(emb.shape) != 2:
                raise ValueError("Domain embedding must have shape (batch_size, embedding_dim). Got shape: {}".format(emb.shape))


        
    @staticmethod
    def identity_forward(x):
        return x
            
    @staticmethod
    def _deactivate_batchnorm_and_dropout(m):
        """
        Disables batchnorm and dropout throughout a model.
        """
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.Dropout):
            m.eval()
            m.forward = GeNVSEncoder.identity_forward

    
    @staticmethod
    def _test():
        """
        Test forward pass.
        """
        encoder = CrossDomainGeNVSEncoder(embedding_dim=69)
        x = torch.randn((1,3,128,128))
        y = encoder(x)
        assert(y.shape == (1,16,64,128,128)), "Test failed. Got shape: {} instead of (1,16,64,128,128)".format(y.shape)
        print("Test without embedding passed!")
        emb = torch.randn((1,69))
        y = encoder(x,emb)
        assert(y.shape == (1,16,64,128,128)), "Test failed. Got shape: {} instead of (1,16,64,128,128)".format(y.shape)
        print("Test with embedding passed!")

#----------------------------------------------------------------------------
# GeNVS DeepLabV3++ Encoder Model.

class ModifiedDLV3PDecoderUpStage(torch.nn.Module):
    """
    Builds a stage of the convolutional layers between each upsampling 
    ConvTranspose2d in the ModifiedDLV3PDecoderUp class.
    """
    def __init__(self, ic, oc, ks, upsample_stage_conv, embedding_dim):
        super().__init__()

        self.first_conv = torch.nn.Conv2d(ic,oc,kernel_size=ks,stride=1,padding="same")

        self.additive_embedding = AdditiveEmbedding(embedding_dim, oc)

        self.additional_conv_layers = torch.nn.ModuleList()
        for i in range(upsample_stage_conv):
            self.additional_conv_layers.append(
                torch.nn.Conv2d(oc,oc,kernel_size=ks,stride=1,padding="same")
            )

        self.relu = torch.nn.ReLU()


    def forward(self, x, emb = None):
        x = self.first_conv(x)
        
        x = self.relu(x)

        for layer in self.additional_conv_layers:
            x = self.additive_embedding(x, emb)
            x = layer(x)
            x = self.relu(x)

        return x



class ModifiedDLV3PDecoderUp(torch.nn.Module):
    """
    Replaces self.up in the DeepLabV3PlusDecoder class.
    Contains learned convolutional layers and skip connections
    from the encoder layers.

    The old up function outputs a tensor with shape (batch_size, 256, 32, 32)
    """
    def __init__(
        self,
        out_channels: int = 256,
        kernel_size: int = 3,
        upsample_stage_conv: int = 2,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.oc = out_channels
        self.ks = kernel_size
        assert(upsample_stage_conv > 0), "upsample_stage_conv must be greater than 0"
        self.upsample_stage_conv = upsample_stage_conv
        self.embedding_dim = embedding_dim
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
        """
        Builds a stage of the convolutional layers between 
        each upsampling ConvTranspose2d.
        """
        return ModifiedDLV3PDecoderUpStage(ic,oc,self.ks,self.upsample_stage_conv,self.embedding_dim)

    def forward(self,aspp_features,features,emb = None):
        """
        Features is a list of outputs from each stage in the resnet encoder.
        """
        # concatenate features[4,5] and aspp_features along the channel dimension
        x = torch.cat([features[4],features[5],aspp_features],dim=1) # (batch_size, 1024, 8, 8)

        # 2x2 conv
        x = self.stage1(x,emb=emb) # (batch_size, 256, 8, 8)
        
        # upconv by 2x with 256 output channels
        x = self.upconv1(x) # (batch_size, 256, 16, 16)

        # concatenate features[3] and x along the channel dimension
        x = torch.cat([features[3],x],dim=1) # (batch_size, 384, 16, 16)

        # 3x3 conv with 256 output channels
        x = self.stage2(x,emb=emb) # (batch_size, 256, 16, 16)

        # upconv by 2x with 256 output channels
        x = self.upconv2(x) # (batch_size, 256, 32, 32)

        # concatenate features[2] and x along the channel dimension
        x = torch.cat([features[2],x],dim=1) # (batch_size, 320, 32, 32)

        # 3x3 conv with 256 output channels
        x = self.stage3(x,emb=emb) # (batch_size, 256, 32, 32)

        return x


class ModifiedDLV3PDecoder(DeepLabV3PlusDecoder):
    """
    Modified to replace self.up with a ModifiedDLV3PDecoderUp.
    The modified upsampler replaces bilinear upsampling with
    learned convolutional layers and skip connections from
    encoder layers.
    """
    def __init__(
        self,
        upsample_conv_kernel: int = 3,
        upsample_stage_conv: int = 2,
        embedding_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)

        # create embedding matching layers
        self.additive_embedding_1 = AdditiveEmbedding(embedding_dim, 512)
        self.additive_embedding_2 = AdditiveEmbedding(embedding_dim, 256)
        self.additive_embedding_3 = AdditiveEmbedding(embedding_dim, 64)
        self.additive_embedding_4 = AdditiveEmbedding(embedding_dim, 304)

        self.modified_up = ModifiedDLV3PDecoderUp(
            out_channels        = self.out_channels,
            kernel_size         = upsample_conv_kernel,
            upsample_stage_conv = upsample_stage_conv,
            embedding_dim  = embedding_dim,
        )

    def forward(self, features, emb = None):
        """
        Features is a list of outputs from each stage in the resnet encoder.
            len(features): 6
            [0] torch.Size([69, 3, 128, 128])
            [1] torch.Size([69, 64, 64, 64])
            [2] torch.Size([69, 64, 32, 32])
            [3] torch.Size([69, 128, 16, 16])
            [4] torch.Size([69, 256, 8, 8])
            [5] torch.Size([69, 512, 8, 8])

        emb is the domain embedding tensor. shape: (batch_size, 1, 128, 128)
        """
        x = self.additive_embedding_1(features[5], emb)
        aspp_features = self.aspp(x) # (batch_size, 256, 8, 8)

        # replace self.up with modified upsampler
        # aspp_features = self.up(aspp_features)
        x = self.additive_embedding_2(aspp_features, emb)
        aspp_features = self.modified_up(x,features,emb=emb)

        x = self.additive_embedding_3(features[2], emb)
        high_res_features = self.block1(x)
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)

        x = self.additive_embedding_4(concat_features, emb)
        fused_features = self.block2(x) # (batch_size, 256, 32, 32)

        return fused_features





class FeatureVolumeUp(ModifiedDLV3PDecoderUp):
    """
    Replaces upsampling that is normally in the segmentation head.
    Contains learned convolutional layers and skip connections
    from the encoder layers.
    """
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

    def forward(self,decoder_output,features,emb = None):
        # upconv by 2x with 256 output channels
        x = self.upconv1(decoder_output) # (batch_size, 256, 64, 64)

        # concatenate features[1] and x along the channel dimension
        x = torch.cat([features[1],x],dim=1) # (batch_size, 320, 64, 64)

        # 3x3 conv with 320 output channels
        x = self.stage1(x,emb=emb) # (batch_size, 256, 64, 64)

        # upconv by 2x with 320 output channels
        x = self.upconv2(x) # (batch_size, 320, 128, 128)

        # concatenate features[0] and x along the channel dimension
        x = torch.cat([features[0],x],dim=1) # (batch_size, 323, 128, 128)

        # 3x3 conv
        x = self.stage2(x,emb=emb) # (batch_size, 256, 128, 128)

        return x

class FeatureVolumeHead(torch.nn.Module):
    """
    Takes the decoder output to produce a feature volume.
    Uses 1x1 convolutions to reshape the upsampled decoder output
    to the desired shape, then reshapes the tensor into a volume.
    """
    def __init__(
        self,
        decoder_channels: int = 256,
        n_features: int = 16,
        depth: int = 64,
        upsample_conv_kernel: int = 3,
        upsample_stage_conv: int = 2,
        encoder_in_channels: int = 3,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.n_features = n_features
        self.depth = depth

        self.up = FeatureVolumeUp(
            encoder_in_channels = encoder_in_channels,
            out_channels        = decoder_channels,
            kernel_size         = upsample_conv_kernel,
            upsample_stage_conv = upsample_stage_conv,
            embedding_dim  = embedding_dim,
        )
        self.reshape_conv = torch.nn.Conv2d( decoder_channels,
                                       n_features*depth,
                                       kernel_size=1,
                                       stride=1,
                                       padding="same"
                                    )
        self.activation = torch.nn.ReLU()
        self.additive_embedding = AdditiveEmbedding(embedding_dim, decoder_channels)

    def forward(self,decoder_output,features,emb = None):
        x = self.up(decoder_output,features,emb=emb)
        x = self.additive_embedding(x, emb)
        x = self.reshape_conv(x)
        x = self.activation(x)
        x = torch.reshape(x,(x.shape[0],self.n_features,self.depth,128,128))
        return x
