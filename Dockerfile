# 1. 젯슨 나노 전용 NVIDIA 공식 이미지 사용 (JetPack 4.6 기준)
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# 2. 컨테이너 내부의 작업 위치 지정
WORKDIR /app

# 3. requirements.txt를 컨테이너 안으로 복사
COPY requirements.txt .

# 4. 의존성 패키지 설치
RUN pip3 install --no-cache-dir -r requirements.txt

# 5. 내 코드를 컨테이너 안으로 모두 복사 (best.pt와 main.py 포함)
COPY . .

# 6. 컨테이너가 실행될 때 자동으로 실행할 명령
CMD ["python3", "Model_track.py"]