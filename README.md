# Welcome to AoP-SAM

We provide three versions of AoP-SAM using ViT-B,ViT-L,ViT-H.

**Note:** And we provide the two-stage training code of AoP-SAM when ViT-B is adopted as the encoder.The codes of ViT-L and ViT-H can be modified by themselves by referring to the codes we provided. Where in addition to modifying the model, you also need to modify the resolution of the input image to 512x512 and *args.encoder_input_size*. for training, we use the following code: *scaler = torch.cuda.amp.GradScaler(enabled=True) andwith torch. autocast(device_type='cuda', dtype=torch.float16, enabled=True)*.