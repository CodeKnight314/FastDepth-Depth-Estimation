# FastDepth-Depth-Estimation
FastDepth model implementation with pytorch on NYU dataset

# FastDepth Model Description (Layer by Layer)

| Layer/Block         | Type                  | Input Channels | Output Channels | Kernel Size | Stride | Padding | Additional Info                   |
|---------------------|-----------------------|----------------|-----------------|-------------|--------|---------|-----------------------------------|
| Initial Conv        | Conv2d + BN + ReLU    | 3              | 32              | 3x3         | 2      | 1       | Encoder initial convolution       |
| Conv Block 1        | Depthwise + Pointwise | 32             | 64              | 3x3, 1x1    | 2, 1   | 1, 0    | Depthwise Separable Convolution    |
| Conv Block 2        | Depthwise + Pointwise | 64             | 128             | 3x3, 1x1    | 1, 1   | 1, 0    | Depthwise Separable Convolution    |
| Conv Block 3        | Depthwise + Pointwise | 128            | 128             | 3x3, 1x1    | 2, 1   | 1, 0    | Depthwise Separable Convolution    |
| Conv Block 4        | Depthwise + Pointwise | 128            | 256             | 3x3, 1x1    | 1, 1   | 1, 0    | Depthwise Separable Convolution    |
| Conv Block 5        | Depthwise + Pointwise | 256            | 256             | 3x3, 1x1    | 2, 1   | 1, 0    | Depthwise Separable Convolution    |
| Conv Block 6        | Depthwise + Pointwise | 256            | 512             | 3x3, 1x1    | 1, 1   | 1, 0    | Depthwise Separable Convolution    |
| Conv Block 7-11     | Depthwise + Pointwise | 512            | 512             | 3x3, 1x1    | 1, 1   | 1, 0    | Repeated depthwise separable block|
| Conv Block 12       | Depthwise + Pointwise | 512            | 1024            | 3x3, 1x1    | 2, 1   | 1, 0    | Depthwise Separable Convolution    |
| Conv Block 13       | Depthwise + Pointwise | 1024           | 1024            | 3x3, 1x1    | 1, 1   | 1, 0    | Depthwise Separable Convolution    |
| Upconv 5            | ConvTranspose2d       | 1024           | 512             | 3x3         | 2      | 1       | Decoder upconvolution             |
| Iconv 5             | Conv2d                | 1024           | 512             | 3x3         | 1      | 1       | Skip connection with encoder      |
| Upconv 4            | ConvTranspose2d       | 512            | 256             | 3x3         | 2      | 1       | Decoder upconvolution             |
| Iconv 4             | Conv2d                | 512            | 256             | 3x3         | 1      | 1       | Skip connection with encoder      |
| Upconv 3            | ConvTranspose2d       | 256            | 128             | 3x3         | 2      | 1       | Decoder upconvolution             |
| Iconv 3             | Conv2d                | 256            | 128             | 3x3         | 1      | 1       | Skip connection with encoder      |
| Upconv 2            | ConvTranspose2d       | 128            | 64              | 3x3         | 2      | 1       | Decoder upconvolution             |
| Iconv 2             | Conv2d                | 128            | 64              | 3x3         | 1      | 1       | Skip connection with encoder      |
| Upconv 1            | ConvTranspose2d       | 64             | 32              | 3x3         | 2      | 1       | Decoder upconvolution             |
| Iconv 1             | Conv2d                | 64             | 32              | 3x3         | 1      | 1       | Skip connection with encoder      |
| Depth Prediction    | Conv2d                | 32             | 1               | 1x1         | 1      | 0       | Final depth prediction layer      |

# Summary
The FastDepth model consists of an encoder based on MobileNet using depthwise separable convolutions, followed by a decoder with transpose convolutions and skip connections to progressively upscale the feature maps and predict depth. Each upsampling layer is concatenated with corresponding feature maps from the encoder, which helps in refining the depth predictions.