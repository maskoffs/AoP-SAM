import torch
import argparse
from models.model_dict import get_model
from fvcore.nn import FlopCountAnalysis, parameter_count_table, ActivationCountAnalysis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('-encoder_input_size', type=int, default=256)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--vit_name', type=str, default='vit_b')

    parser.add_argument('--sam_ckpt', type=str,
                        default="./checkpoints/sam_vit_b_01ec64.pth",
                        help='Pretrained checkpoint of SAM')

    args = parser.parse_args()
    model = get_model(args=args)


    x = torch.randn(1, 3, 256, 256)
    flops = FlopCountAnalysis(model, x)
    print(flops.total())
    print(parameter_count_table(model))

    outPack = model(x)
    print(outPack['low_res_logits'].shape)
    print(outPack['masks'].shape)  # masks通过对low_res_logits插值得到。