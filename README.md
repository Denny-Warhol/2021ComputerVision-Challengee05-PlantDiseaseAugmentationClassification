# 2021ComputerVision-Challengee14-Plant Disease Augmentation Classification

## 환경 구축

- Conda를 사용해 가상 환경을 구축하고 필요한 라이브러리를 설치함
- TensorFlow version: 2.6

```
conda create --name tf2_6 python=3.6
pip install --upgrade tensorflow
pip install pandas
```

## Seed 고정

실험을 진행하는 과정에서 똑같은 설정으로 코드를 실행할 때 동일한 결과를 얻기 위해 모델을 훈련하기 앞서 제공하고 있는 코드에 아래와 같이 seed를 고정하는 코드를 추가함

```python
seed_value = 20
print("Train with random seed: ", seed_value)

import warnings

warnings.filterwarnings('ignore')

import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)
from keras import backend as K
```

##  Epoch 및 훈련 설정 수정

Overfitting 및 코드 에러를 해결하기 위해 제공하고 있는 고드를 아래와 같이 수정하고 모델 훈련을 진행함

```python
# epochs = 150
epochs = 5
history = model.fit_generator(
    trainGen, 
    epochs=epochs,
#     steps_per_epoch=trainGen.samples / epochs, 
    validation_data=validationGen,
#     validation_steps=trainGen.samples / epochs,
)
```

## Leader board 업로드 결과

훈련한 모델을 사용해 예측을 진행한 결과를 Leader board에 업로드하여 성능을 측정했을 때 baseline 성능과 동일한 53%의 정확도를 기록함

![Leader board test resultf](D:\수업\212\컴퓨터비전\challenge01_Plant Disease Augmentation Classification\ComputerVisionProject1-master\images\leadboard_result1.png)
