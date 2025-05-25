# Welcome to AoP-SAM

We provide three versions of AoP-SAM using ViT-B,ViT-L,ViT-H.



**Note:** And we provide the **two-stage training code** of AoP-SAM when ViT-B is adopted as the encoder. The codes of ViT-L and ViT-H can be modified by themselves by referring to the codes we provided. Where in addition to modifying the model, you also need to modify the resolution of the input image to 512x512 and *args.encoder_input_size*. for training, we use the following code: 

```python
scaler = torch.cuda.amp.GradScaler(enabled=True) 
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
```



#### Model Weight

Model weights have been uploaded to **10.5281/zenodo.15509422**. You can download the weights of AoP-SAM using ViT-B, ViT-L, and ViT-H as encoders.



#### Citation

```
Zhou, Z., Lu, Y., Bai, J., Campello, V. M., Feng, F., & Lekadir, K. (2025). Segment anything model for fetal head-pubic symphysis segmentation in intrapartum ultrasound image analysis. *Expert Systems with Applications*, *263*, 125699.
```

