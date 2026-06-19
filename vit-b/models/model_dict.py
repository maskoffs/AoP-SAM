from models.segment_anything_aopsam.build_aopsam import aopsam_model_registry


def get_model(args=None):
    model = aopsam_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)

    return model
