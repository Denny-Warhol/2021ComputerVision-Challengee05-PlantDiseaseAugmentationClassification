# ComputerVisionProject1

병해충을 탐지하여 제거하는 작업은 작물의 생산량과 품질과 깊은 관련이 있으므로 병해충을 조기에 발견하여 조치하는 것으로 피해를 줄일 수 있다. 
이를 위해서 정상적인 작물과 병든 작물을 구분하는 기술이 필요하다. 



![그림1](https://user-images.githubusercontent.com/85859441/137726532-cf405587-6a69-4ce0-a3b2-87fbc88cfa9f.png)

병해충 증강 및 학습 흐름도 [1]

영상 : https://youtu.be/DA_cXup95GI

## 데이터셋

![캡처](https://user-images.githubusercontent.com/85859441/137592829-909c6077-5207-4235-bb15-2c773bcde151.PNG)

AI Hub와 농진청에서 제공하는 화상병, 건강한 잎, leafspot 이미지를 이용했습니다.
데이터 셋은 vision 폴더에 첨부했습니다


## 테스트 제출 자료

csv 형식의 파일 경로 + 파일 이름과 예측 라벨이 포함된 파일 (Name 컬럼은 열의 값이 사진과 동일해야합니다)

예시 사진) 제출 자료

![캡처](https://user-images.githubusercontent.com/85859441/137592622-858fa11f-682d-4776-861b-346515f5c931.PNG)

라벨은 화상병, 건강한 잎, Leafspot 순으로 각각 0,1,2 입니다 

평가 지표는 라벨에 대한 정확도입니다.


## 베이스라인

본 과제는 '딥러닝 기반 표고버섯 병해충 이미지 분석에 관한 연구'를 베이스라인으로 합니다.
[1] https://www.koreascience.or.kr/article/JAKO202010163508707.pdf

코드 베이스라인 : https://codingcrews.github.io/2019/01/19/cat-dog-resnet/ 


## 추천 논문

Effective Data Augmentation and Training Techniques for Improving Deep Learning in Plant Leaf Disease Recognition
https://www.researchgate.net/publication/346607345_Effective_Data_Augmentation_and_Training_Techniques_for_Improving_Deep_Learning_in_Plant_Leaf_Disease_Recognition


Unsupervised image translation using adversarial networks for improved plant disease recognition, Computers and Electronics in Agriculture
https://www.sciencedirect.com/science/article/pii/S0168169919315339




