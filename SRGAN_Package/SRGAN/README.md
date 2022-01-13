# Super Resolution Image (SRGAN)

Generate **S R G A N ** Image

## Instructions

1.Install:

```
pip install SuperResolution-GANs
pip install keras==2.3.1
pip install tensorflow_gpu==2.1.0
pip install h5py==2.10.0 
```


2. Download Our Model
```python
import gdown
url = 'https://drive.google.com/uc?id=116fpSp3dUBtH7GkCoZ4UymK76xRTrZLs'
output = 'model.h5'
gdown.download(url, output, quiet=False)
```

3.Generate Super Resolution Image:
```python
from super_resolution_gans import srgan_utils
srgan_utils.SRGAN_predict(lr_image_path, model_path , outputPath)
```
4.show Super Resolution image::
```python
#Original Image (Low Resolution)
show_image(LR_img)
#Super Resolution Image
show_image(SR_img)
```

