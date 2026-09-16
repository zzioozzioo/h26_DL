# YOLO Mask Detection

Stable Diffusion으로 생성한 마스크 착용 이미지를 기반으로 YOLO 학습 데이터를 구축하고, YOLO 모델을 활용하여 이미지 내 사람의 **마스크 착용 여부를 탐지하는 딥러닝 프로젝트**

---

## 📌 1. 프로젝트 개요

* **프로젝트 주제:** YOLO 기반 마스크 착용 여부 탐지
* **개발 환경:** Python 3.11
* **주요 역할:** 데이터 생성, 얼굴 검출, YOLO 데이터셋 구축 및 모델 학습, 실제 이미지 탐지
* **GPU:** CUDA 환경 기반 딥러닝 학습 및 추론

#### 💡 프로젝트 목표

* 다양한 환경과 인물의 마스크 착용 이미지를 생성하여 학습 데이터 확보
* RetinaFace를 활용하여 이미지 내 얼굴 영역을 검출하고 YOLO 학습 데이터 구축
* YOLO 객체 탐지 모델을 학습하여 `mask_on`, `no_mask` 두 가지 클래스를 구분
* 학습된 모델을 실제 이미지에 적용하여 마스크 착용 여부를 탐지

---

## 🧠 2. 프로젝트 Pipeline

```text
Stable Diffusion
       │
       ▼
마스크 착용 이미지 생성
       │
       ▼
RetinaFace
얼굴 영역 검출
       │
       ▼
YOLO 학습 데이터 구축
       │
       ▼
YOLO 모델 학습
       │
       ▼
Mask Detection
       │
       ▼
실제 이미지에서
마스크 착용 여부 탐지
```

---

## <p>📚 3. STACKS

<div>

### Environment

<img src="https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
<img src="https://img.shields.io/badge/Anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white">
<img src="https://img.shields.io/badge/Git-F03C2E?style=for-the-badge&logo=git&logoColor=white">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">

### Development

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">

### Deep Learning

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/YOLO-111111?style=for-the-badge&logo=yolo&logoColor=white">
<img src="https://img.shields.io/badge/Ultralytics-111111?style=for-the-badge&logo=ultralytics&logoColor=white">
<img src="https://img.shields.io/badge/Stable%20Diffusion-412991?style=for-the-badge&logo=stabilityai&logoColor=white">

### Computer Vision

<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/RetinaFace-4285F4?style=for-the-badge">

### Data & Utilities

<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white">
<img src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white">

<br>
</div>

---

## 🎯 4. 주요 기능

#### **🖼️ Stable Diffusion 기반 학습 이미지 생성**

* Stable Diffusion의 `Lykon/dreamshaper-8` 모델을 활용하여 마스크 착용 인물 이미지 생성
* 이미지 생성 과정에서 인물의 성별, 포즈, 배경, 의상, 마스크 색상, 표정, 촬영 구도 등의 조건을 조합
* 다양한 조건의 이미지를 생성하여 데이터셋의 다양성을 확보
* `torch.float16`, CUDA, xFormers 등을 활용하여 GPU 기반 이미지 생성 및 메모리 효율 개선

#### **👤 RetinaFace 기반 얼굴 검출**

* RetinaFace와 ResNet50 Backbone을 활용하여 이미지 내 얼굴 영역 탐지
* 사전 학습된 `Resnet50_Final.pth` 가중치를 사용하여 얼굴 검출 수행
* `PriorBox`, `decode`, `NMS` 등을 활용하여 얼굴 위치를 계산하고 중복 검출 영역 제거
* 검출된 얼굴 영역을 기반으로 YOLO 학습 데이터 구축

#### **🏷️ YOLO 학습 데이터셋 구축**

* 얼굴 검출 결과를 YOLO 형식의 학습 데이터로 변환
* `train` / `val` 데이터셋으로 분리하여 모델 학습 및 검증
* 마스크 착용 여부를 다음 두 개의 클래스로 정의

| Class ID | Class     |
| :------: | :-------- |
|    `0`   | `mask_on` |
|    `1`   | `no_mask` |

* 데이터셋의 클래스 정보는 `data.yaml`을 통해 관리

#### **🤖 YOLO 기반 마스크 착용 여부 탐지**

* Ultralytics YOLO를 활용하여 `mask_on`, `no_mask` 객체 탐지 모델 학습
* 학습 과정에서 구축한 마스크 데이터셋을 사용하여 모델의 마스크 착용 여부 분류 및 위치 탐지 수행
* 학습 결과로 생성된 `best.pt` 가중치를 활용하여 실제 이미지에 대한 추론 수행

#### **📷 실제 이미지 탐지**

* 학습된 YOLO 모델을 실제 이미지에 적용
* 이미지 내 사람의 마스크 착용 여부와 위치를 Bounding Box 형태로 탐지
* `mask_on`과 `no_mask` 클래스를 구분하여 객체 탐지 결과 출력
* 실제 검증 과정에서 다수의 이미지에 대해 YOLO 추론 수행

---

## 🔄 5. 데이터 생성 및 학습 과정

### Step 1. 마스크 이미지 생성

Stable Diffusion을 이용하여 마스크를 착용한 사람의 이미지를 생성한다.

```text
인물 조건
 ├─ Gender
 ├─ Pose
 ├─ Background
 ├─ Clothes
 ├─ Mask Color
 ├─ Expression
 └─ Shot Type
```

이미지 생성 크기는 `640 × 640`으로 설정하고, Batch 단위로 이미지를 생성한다.

---

### Step 2. 얼굴 영역 검출

생성된 이미지에서 RetinaFace를 사용하여 얼굴 영역을 검출한다.

```text
Input Image
     │
     ▼
RetinaFace
     │
     ├── Face Bounding Box
     │
     ▼
Detection Result
```

RetinaFace 모델에는 ResNet50 기반 설정(`cfg_re50`)과 사전 학습된 가중치를 적용한다.

---

### Step 3. YOLO 데이터셋 구성

얼굴 검출 결과를 바탕으로 YOLO 학습에 사용할 이미지와 Label 데이터를 구성한다.

```text
mask_yolo_dataset/
│
├─ images/
│  ├─ train/
│  └─ val/
│
├─ labels/
│  ├─ train/
│  └─ val/
│
└─ data.yaml
```

`data.yaml`에서는 학습 데이터와 검증 데이터의 경로 및 클래스 정보를 정의한다.

---

### Step 4. YOLO 모델 학습

구축한 데이터셋을 이용하여 YOLO 모델을 학습한다.

```text
YOLO Dataset
     │
     ▼
Model Training
     │
     ├── mask_on
     └── no_mask
     │
     ▼
best.pt
```

학습 과정에서 생성된 `best.pt`를 최종 모델 가중치로 사용하여 실제 이미지에 대한 탐지를 수행한다.

---

## 📂 6. 폴더 구조

```text
yolo/
│
├─ 01_make_masked_man_test.ipynb
│   # Stable Diffusion 환경 구성 및 테스트 이미지 생성
│
├─ 02_make_masked_man.ipynb
│   # 다양한 조건을 조합한 마스크 이미지 데이터 생성
│
├─ 03_face_detection_retinaface.ipynb
│   # RetinaFace를 활용한 얼굴 검출
│
├─ 04_make_yolo_train_data_p35_p62.ipynb
│   # 얼굴 검출 결과를 활용한 YOLO 학습 데이터 구축
│
├─ 05_yolo_train_p74_p90.ipynb
│   # YOLO 모델 학습
│
├─ 06_real_detect_p103_p107.ipynb
│   # 학습된 YOLO 모델을 이용한 실제 이미지 탐지
│
├─ mask_yolo_dataset/
│   │
│   ├─ images/
│   │   ├─ train/
│   │   └─ val/
│   │
│   ├─ labels/
│   │   ├─ train/
│   │   └─ val/
│   │
│   └─ data.yaml
│
└─ README.md
```

현재 저장소에는 위와 같은 순서의 6개 Jupyter Notebook과 `mask_yolo_dataset` 디렉터리가 구성되어 있다.

---

## ⚙️ 7. 주요 환경 설정

### Python

```text
Python 3.11.16
```

### PyTorch

```text
PyTorch 2.4.0
CUDA 11.8
```

프로젝트 환경에서는 CUDA 사용 여부를 확인한 후 GPU를 사용하여 RetinaFace 및 Stable Diffusion 모델을 실행한다.

### 주요 라이브러리

```text
torch
torchvision
ultralytics
diffusers
transformers
accelerate
opencv-python
numpy
matplotlib
Pillow
xformers
```

---

## 🚀 8. 실행 순서

각 Notebook은 다음 순서로 실행한다.

### 1. 데이터 생성

```text
01_make_masked_man_test.ipynb
        ↓
02_make_masked_man.ipynb
```

Stable Diffusion 환경을 구성한 후 마스크 관련 이미지를 생성한다.

### 2. 얼굴 검출

```text
03_face_detection_retinaface.ipynb
```

RetinaFace를 이용하여 이미지 내 얼굴 영역을 검출한다.

### 3. YOLO 데이터셋 생성

```text
04_make_yolo_train_data_p35_p62.ipynb
```

얼굴 검출 결과를 YOLO 학습 형식으로 변환한다.

### 4. YOLO 학습

```text
05_yolo_train_p74_p90.ipynb
```

구축한 데이터셋을 이용하여 마스크 착용 여부 탐지 모델을 학습한다.

### 5. 실제 이미지 탐지

```text
06_real_detect_p103_p107.ipynb
```

학습된 `best.pt` 모델을 불러와 실제 이미지에서 마스크 착용 여부를 탐지한다.

---

## 📌 9. 데이터셋 클래스

본 프로젝트에서는 마스크 착용 여부를 2개의 클래스로 분류한다.

```yaml
names:
  0: mask_on
  1: no_mask

nc: 2
```

|   Class   | 설명              |
| :-------: | :-------------- |
| `mask_on` | 마스크를 착용한 얼굴     |
| `no_mask` | 마스크를 착용하지 않은 얼굴 |

---

## 💡 10. 프로젝트를 통해 학습한 내용

* 생성형 AI를 활용한 딥러닝 학습 데이터 구축
* Stable Diffusion을 활용한 이미지 생성 및 Prompt 설계
* RetinaFace를 활용한 얼굴 검출
* Bounding Box 기반 객체 탐지 데이터 구축
* YOLO 데이터셋 구조 및 `data.yaml` 구성
* Ultralytics YOLO 모델 학습 및 추론
* CUDA를 활용한 GPU 기반 딥러닝 모델 실행
* 실제 이미지에 대한 객체 탐지 결과 확인
