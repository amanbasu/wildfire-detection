def trans_unet(image_size, channels=3, blocks=8, embeddings=512, classes=1):
    from self_attention_cv.transunet import TransUnet
    '''
    borrowed TransUnet implementation from -
    https://github.com/The-AI-Summer/self-attention-cv
    '''
    return TransUnet(in_channels=channels,
                        img_dim=image_size,
                        classes=classes,
                    )

def trans_unet_orig(image_size, n_classes=1):
    from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
    from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    import numpy as np
    '''
    borrowed from https://github.com/Beckschen/TransUNet
    '''
    vit_name = 'ViT-B_16'
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 0

    net = ViT_seg(config_vit, img_size=image_size, num_classes=config_vit.n_classes)
    net.load_from(weights=np.load(config_vit.pretrained_path))
    return net

def swin_unet(image_size, batch_size, n_classes=1, pretrain=False):
    import sys
    sys.path.append('Swin-Unet/')
    from config import get_config
    from networks.vision_transformer import SwinUnet as ViT_seg

    '''
    borrowed from https://github.com/HuCaoFighting/Swin-Unet
    '''
    args = {'batch_size': batch_size, 'pretrain': pretrain}
    config = get_config(args)
    model = ViT_seg(config, img_size=image_size, num_classes=n_classes)
    model.load_from(config)
    return model

def unet(n_channels=3, n_classes=1):
    '''
    Our own implementation of UNet
    '''
    from UNet import UNet
    return UNet(n_channels, n_classes)