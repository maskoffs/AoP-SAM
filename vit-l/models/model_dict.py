from models.segment_anything_aopsam.build_aopsam import aopsam_model_registry
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table, ActivationCountAnalysis


def get_model(args):
    model = aopsam_model_registry['vit_l'](args=args, checkpoint=args.sam_ckpt)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Networks')

    parser.add_argument('-encoder_input_size', type=int, default=512)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--vit_name', type=str, default='vit_l')
    parser.add_argument('--sam_ckpt', type=str,
                        default="../checkpoints/sam_vit_l_0b3195.pth")

    args = parser.parse_args()
    model = get_model(args=args)

    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, ActivationCountAnalysis

    x = torch.randn(1, 3, 512, 512)
    flops = FlopCountAnalysis(model, x)
    print(flops.total())
    print(parameter_count_table(model))
    outPack = model(x)
    print(outPack['low_res_logits'].shape)
    print(outPack['masks'].shape)
