시험 입력 파일: starlab_test.sh

시험 출력 파일: 콘솔 및 vts (numpy로 저장된 text 파일)\
진행도 및 시험 결과는 콘솔에 출력된다.\
실험 1의 4가지 표현 방식은 experiments/functional_correspondence/test안에 4개의 디렉토리를 만들어 저장한다.

시험 내용\
실험 1) 3차원 모델 일치도 측정 형식 개발\
두 개의 모델 사이 대응 되는 점의 정보와, 두 개의 모델을 basis로 변환하여 대응시킨 functional map의 정보를 총 4가지 다른 형태로 저장하였다.\
Injective Correspondence: 모델 2의 정점 인덱스(index)를 모델 1의 정점 인덱스로 대응시킨 값 (label파일), 그리고 실제 점 인덱스를 대입해서 매핑한 모델 1의 점 값을 저장하였다.\
Bijective Correspondence: 모델 2의 정점 인덱스를 모델 1의 모든 인덱스로 대응시킨 값 (label파일), 그리고 실제 점 인덱스를 대입해서 매핑한 모델 1의 점 값을 저장하였다.\
Probabilistic Correspondence: 두 모델 사이의 basis의 대응을 행렬로 나타내고, 모델 1의 basis가 모델 2의 basis로 매핑될 확률과 모델 2의 basis가 모델 1의 basis로 매핑될 확률을 저장하였다.\
Binary Correspondence: 두 모델 사이의 basis의 대응을 행렬로 나타낸 사이에서 행렬의 절댓값에 대해 threshold(임계값)을 0.11로 지정하여 초과일때 1, 넘지 못하면 0으로 저장하였다.

실험 2) 3차원 모델 일치도 측지 오차 5% 이내\
Ground truth 측지 거리와 새로 계산한 모델간의 점 끼리의 측지 거리 사이의 오차를 계산하였다.\
측지 거리는normalize(정규화) 시킨 후 100을 곱하여 퍼센트로 나타내었으며, 총 190개 모델 pair(쌍)의 측지 오차값의 평균이 5프로 이내가 되는지 확인하였다.

실험 3) 3차원 모델 일치도 측정율 90% 이상\
Princeton Shape Benchmark를 리메싱 하여 총 210개의 모델을 사용하였다.\
임의의 모델들을(기본 10개) 뽑아 모델 쌍을 만들어 오차를 측정하고, 일치도가 높은 모델을 뽑는다.\
검출율이 90프로 이상이 되는지 확인하였다.

시스템에 미리 설치해야 할것:\
모든 절차는 모델 대응 탐색 코드 실행 방법 참고\
도커 이미지와 파이썬, 토치(Torch)를 설치해야 한다. 도커 이미지는 Cuda 11.1, 운영체제 Ubuntu20.04를 지원한다.\
도커 이미지: nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04\
도커 파일: https://drive.google.com/file/d/1QsGliMdCpD1HTeP2VIKVMNoINXStLV4P/view?usp=sharing


실행 환경:\
GPU버전: NVIDIA GeForce RTX 3090 Ti\
운영체제: ubuntu20.04\
언어 버전: Python 3.8.10\
Cuda 버전: 11.1\
Torch 버전: 1.8.1\
Torchvision 버전: 0.9.1

실행 절차:
1. 도커를 킨다.\
docker start difnet11.1[지정한 도커의 이름]\
docker attach difnet11.1[지정한 도커의 이름]\
su - staruser[사용자 이름]\
2. 소스코드 workspace/func_corr으로 들어간다.\
cd ../..\
cd workspace/func_corr
3. 아래 명령어를 입력하면 실험이 실행된다.\
bash starlab_test.sh
