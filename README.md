# Helmet detection(YOLOv8)
## 1. 개발환경
 ### 1) 사용한 OS
 #### - MacOS, Window 10
 ### 2) 파이썬 버전
 #### - 3.10.12
 ### 3) 설치한 라이브러리
 #### - PyYAML, ultralytics
 ## 2. 실행방법
 ### 1) 리포지토리에 업로드 한 Yolo.ipynb파일을 구글 colab에서 열기
 ### 2) 커스텀 데이터를 colab으로 다운로드
  ```python
 !wget -O Helmet_data.zip https://app.roboflow.com/ds/1Rj6BECRju?key=exdDCuJd4R
 ```
```python
import zipfile

with zipfile.ZipFile('/content/Helmet_data.zip') as target_file:
    target_file.extractall('/content/Helmet_Data')
```
### 3) 헬멧 데이터에 맞는 YAML파일 생성
```python
!pip install PyYAML
```
```python
import yaml

data = { 'train' : '/content/Helmet_Data/train/images/',
        'val' : '/content/Helmet_Data/valid/images/',
         'test' : '/content/Helmet_Data/test/images',
         'names' : ['Helmet', 'NoHelmet'],
         'nc' : 2}

with open('/content/Helmet_Data/Helmet_Data.yaml', 'w') as f: 
    yaml.dump(data,f)

with open('/content/Helmet_Data/Helmet_Data.yaml', 'r') as f: 
    helmet_yaml = yaml.safe_load(f)
    display(helmet_yaml)
```
### 4) YOLOv8 설치
```python
!pip install ultralytics
```
```python
import ultralytics

ultralytics.checks()
```
### 5) 전처리 모델 로드
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```
### 6) YOLOv8 커스텀 데이터 학습하기
```python
model.train(data = '/content/Helmet_Data/Helmet_Data.yaml', epochs=100, patience=25, batch=32, imgsz=320)
```
### 7) 학습된 YOLOv8 이용해서 테스트 이미지 예측
```python
results = model.predict(source='/content/Helmet_Data/test/images/', save=True)
```

# Helmet detection(MoblieNetV3)
## 1. 개발환경
 ### 1) 사용한 OS
 #### - posix
 ### 2) 파이썬 버전
 #### - 3.10.12
 ### 3) 설치한 라이브러리
 #### - os, json, cv2, numpy, tensorflow, keras
 ## 2. 실행방법
 ### 1) 데이터 로드 및 전처리
 ```python
!pip install tensorflow opencv-python
```
### 2) 라이브러리 선언
```python
import os
import json
import cv2
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow as tf
```
### 3) 데이터 전처리
```python
# 케라스 세션 초기화
tf.keras.backend.clear_session()

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

import os
import json
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"에러: 이미지를 불러올 수 없습니다. 이미지 경로: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image.astype(np.float32)

# 데이터 Generator 함수
def data_generator(annotation_file, image_directory, batch_size=32):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images_paths = []
    labels = []

    for entry in data["annotations"]:
        image_id = entry["image_id"]
        image_filename = data["images"][image_id]["file_name"]
        image_path = os.path.join(image_directory, image_filename)

        images_paths.append(image_path)
        labels.append(entry["category_id"])

    num_samples = len(images_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_labels = []
            batch_paths = images_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]

            for image_path in batch_paths:
                image = load_and_preprocess_image(image_path)
                if image is not None:
                    batch_images.append(image)

            if batch_images:
                batch_images = np.array(batch_images)
                batch_labels = to_categorical(batch_labels, num_classes=len(data["categories"]))
                yield batch_images, batch_labels

# 데이터 로딩 및 전처리
annotation_file = "/content/drive/MyDrive/DataV9/train/_annotations.coco.json"
image_directory = "/content/drive/MyDrive/DataV9/train/train"
batch_size = 32

data_gen = data_generator(annotation_file, image_directory, batch_size=batch_size)
```
### 4) 모델 정의
```python
from tensorflow.keras.layers import GlobalAveragePooling2D

# 데이터 로딩 및 전처리 함수
def load_data(annotation_file, image_directory, batch_size=32):
    return images, labels  # images와 labels 반환

# 데이터 로딩
annotation_file = "/content/drive/MyDrive/DataV9/train/_annotations.coco.json" # 이미지 경로는 개인에 맞춰 수정
image_directory = "/content/drive/MyDrive/DataV9/train/train"
batch_size = 10

images, labels = load_data(annotation_file, image_directory, batch_size=batch_size)

# 모델 구성
base_model = MobileNetV3Small(weights='imagenet',
                              include_top=False,
                              input_shape=(224, 224, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())  # 메모리 효율을 위해 Global Average Pooling 레이어 사용
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 텐서보드 로그 디렉토리 지정
log_dir = "logs/weights_visualization"

# 텐서보드 콜백 설정
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

import numpy as np
from tensorflow.keras.utils import to_categorical

# 이미지 데이터를 numpy 배열로 변환
images = np.array(images)
# 라벨을 one-hot 인코딩 된 텐서 형태로 변환
labels = to_categorical(labels, num_classes=len(coco_data["categories"]))
```
### 5) 모델 학습
```python
# 모델 학습
num_epochs = 70
batch_size = 10  # 메모리 효율을 위해 작은 배치 크기 사용
model.fit(images, labels, epochs=num_epochs, batch_size=batch_size, callbacks=[tensorboard_callback])
```
### 6) 모델 저장
```python
# 모델 저장
model.save("/content/drive/MyDrive/saved_model/trained_model(MobileNetV3).h5")
```

### 7) 텐서보드 모델학습현황 확인
```python
%load_ext tensorboard
%tensorboard --logdir logs/weights_visualization  # 학습 로그가 있는 디렉토리 경로 지정
```
### 8) 모델 분류 테스트
```python
# 모델이 분류하고있는지 정상작동 확인
debug_images = images[:15] # 2개 이미지 선택. 수정가능
debug_predictions = model.predict(debug_images)

for i in range(len(debug_images)):
    true_label = np.argmax(labels[i])
    predicted_label = np.argmax(debug_predictions[i])
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}, Probabilities: {debug_predictions[i]}")
```
### 9) 테스트
```python
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from google.colab.patches import cv2_imshow
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 테스트 데이터 JSON 파일 로드
with open("/content/drive/MyDrive/DataV9/test/_annotations.coco.json", 'r') as f:
    test_coco_data = json.load(f)

# 테스트 이미지 경로
test_image_directory = "/content/drive/MyDrive/DataV9/test/test"

# 저장된 모델 로드 (MobileNetV3)
loaded_model = load_model("/content/drive/MyDrive/saved_model/trained_model(MobileNetV3).h5")

# JSON 파일 내 categories에서 클래스 라벨 정보 추출
# 클래스 레이블 0 은 분류에 무의미한 정보이므로 제외
class_labels_json = {
    category["id"]: category["name"]
    for category in test_coco_data["categories"]
    if category["id"] != 0  # "id"가 0인 경우는 제외
}

# 정확도 계산을 위한 변수 초기화
total_images = len(test_coco_data["images"])
correct_predictions = 0

# 테스트 이미지에 대한 예측과 정확도 계산
for entry in test_coco_data["annotations"]:
    image_id = entry["image_id"]
    image_filename = test_coco_data["images"][image_id]["file_name"]
    image_path = os.path.join(test_image_directory, image_filename)

    # 이미지 로드 및 전처리
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"에러: 이미지를 불러올 수 없습니다. 이미지 경로: {image_path}")
        continue

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(original_image, (224, 224))
    image = preprocess_input(image.astype(np.float32))

    # 모델 예측
    predictions = loaded_model.predict(np.expand_dims(image, axis=0))[0]

    # 예측 결과와 실제 레이블 비교하여 정확도 확인
    true_label = entry["category_id"]
    predicted_label = np.argmax(predictions)

    # 클래스 라벨 가져오기
    true_label_name = class_labels_json[true_label]
    predicted_label_name = class_labels_json[predicted_label]

    # 0에 해당하는 클래스 라벨 제외하고 출력
    if true_label != 0:
        print(f"True Label: {true_label_name}, Predicted Label: {predicted_label_name}")

        # 바운딩 박스 차이 및 신뢰도 출력
        for label_id, prob in enumerate(predictions):
            if label_id != 0:  # 클래스 라벨이 0이 아닌 경우에만 출력
                label_name = class_labels_json.get(label_id, "Unknown")
                print(f"{label_name}: {prob * 100:.2f}%")

        # 예측에 실패한 경우 이미지 출력
        if true_label != predicted_label:
            print("예측 실패한 이미지 출력")
            cv2_imshow(original_image)
```
# Helmet detection(ResNet152)
## 1. 개발환경
 ### 1) 사용한 OS
 #### - posix
 ### 2) 파이썬 버전
 #### - 3.10.12
 ### 3) 설치한 라이브러리
 #### - os, json, cv2, numpy, tensorflow
 ## 2. 실행방법
 ### 1) 데이터 로드 및 전처리
```python
!pip install tensorflow opencv-python
```
### 2) 라이브러리 선언
```python
import os
import json
import cv2
import numpy as np
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# RestNet50
# 모델 변경시 사용
import os
import json
import cv2
import numpy as np
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
```
### 3) 데이터 전처리
```python
# train JSON 파일 로드
with open("/content/drive/MyDrive/DataV9/train/_annotations.coco.json", 'r') as f: # 모든 데이터 경로는 개인에 맞춰 수정
    coco_data = json.load(f)

# train 이미지 디렉토리 경로
image_directory = "/content/drive/MyDrive/DataV9/train/train"

# 이미지 및 라벨 데이터 로딩 및 전처리
images = []
labels = []

for entry in coco_data["annotations"]:
    image_id = entry["image_id"]
    image_filename = coco_data["images"][image_id]["file_name"]
    image_path = os.path.join(image_directory, image_filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"에러: 이미지 {image_id}를 불러올 수 없습니다. 이미지 경로: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image.astype(np.float32))

    label = entry["category_id"]
    labels.append(label)
    images.append(image)

# images를 numpy 배열로 변환하여 저장
np.save('saved_images.npy', np.array(images))

images = np.array(images)
labels = to_categorical(labels, num_classes=len(coco_data["categories"]))
```
### 4) 모델 정의
```python
# 텐서보드를 활용한 학습과정 모니터링
import tensorflow as tf

# Define ResNet152
base_model = ResNet152(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

##### Define ResNet-50 #####

#base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#for layer in base_model.layers:
#    layer.trainable = False

##### Define ResNet-50 #####

# 모델 구성
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(coco_data["categories"]), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 텐서보드 로그 디렉토리 지정
log_dir = "logs/weights_visualization"

# 텐서보드 콜백 설정
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```
### 5) 모델 학습
```python
# 모델 학습
num_epochs = 200
batch_size = 50
model.fit(images, labels, epochs=num_epochs, batch_size=batch_size, callbacks=[tensorboard_callback])
```

### 6) 모델 저장
```pyhton
# 모델 저장
model.save("/content/drive/MyDrive/saved_model/trained_model(resnet152).h5")
```

### 7) 텐서보드 모델학습현황 확인
```python
%load_ext tensorboard
%tensorboard --logdir logs/weights_visualization  # 학습 로그가 있는 디렉토리 경로 지정
```

### 8) 모델 분류테스트기
```python
# 모델이 분류하고있는지 정상작동 확인
debug_images = images[:2] # 2개 이미지 선택. 수정가능
debug_predictions = model.predict(debug_images)

for i in range(len(debug_images)):
    true_label = np.argmax(labels[i])
    predicted_label = np.argmax(debug_predictions[i])
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}, Probabilities: {debug_predictions[i]}")
```

### 9) 테스트
```python
from google.colab.patches import cv2_imshow
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 테스트 데이터셋 JSON 파일 로드
with open("/content/drive/MyDrive/data_forResnet/test/_annotations.coco.json", 'r') as f:
    test_coco_data = json.load(f)

# 테스트 이미지 디렉토리 경로
test_image_directory = "/content/drive/MyDrive/data_forResnet/test/test/"

# 저장된 모델 로드
loaded_model = load_model("/content/drive/MyDrive/saved_model/trained_model.h5")

# JSON 파일 내 categories에서 클래스 라벨 정보 추출
class_labels_json = {
    category["id"]: category["name"]
    for category in test_coco_data["categories"]
    if category["id"] != 0  # "id"가 0인 경우는 제외
}

# 정확도 계산을 위한 변수 초기화
total_images = len(test_coco_data["images"])
correct_predictions = 0

# 테스트 이미지에 대한 예측과 정확도 계산
for entry in test_coco_data["annotations"]:
    image_id = entry["image_id"]
    image_filename = test_coco_data["images"][image_id]["file_name"]
    image_path = test_image_directory + image_filename

    # 이미지 로드 및 전처리
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(original_image, (224, 224))
    image = preprocess_input(image.astype(np.float32))

    # 모델 예측
    predictions = loaded_model.predict(np.expand_dims(image, axis=0))[0]

    # 예측 결과와 실제 레이블 비교하여 정확도 확인
    true_label = entry["category_id"]
    predicted_label = np.argmax(predictions)

    # 클래스 라벨 가져오기
    true_label_name = class_labels_json[true_label]
    predicted_label_name = class_labels_json[predicted_label]

    # 0에 해당하는 클래스 라벨 제외하고 출력
    if true_label != 0:
        print(f"True Label: {true_label_name}, Predicted Label: {predicted_label_name}")

        # 바운딩 박스 차이 및 신뢰도 출력
        for label_id, prob in enumerate(predictions):
            if label_id != 0:  # 클래스 라벨이 0이 아닌 경우에만 출력
                label_name = class_labels_json.get(label_id, "Unknown")
                print(f"{label_name}: {prob * 100:.2f}%")

        # 예측에 실패한 경우 이미지 출력
        if true_label != predicted_label:
            print("예측 실패한 이미지 출력")
            cv2_imshow(original_image)
```

