# from modeling.backbone import resnet, xception, drn, mobilenet
from model.backbone import mobilenet

def build_backbone(backbone, output_stride, BatchNorm):
    # if backbone == 'resnet':
    #     return resnet.ResNet101(output_stride, BatchNorm)
    # elif backbone == 'xception':
    #     return xception.AlignedXception(output_stride, BatchNorm)
    # elif backbone == 'drn':
    #     return drn.drn_d_54(BatchNorm)
    if backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError