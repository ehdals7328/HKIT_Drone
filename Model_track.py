import cv2
import time
from ultralytics import YOLO

# 1. YOLO Nano 모델 로드
model = YOLO('best.pt')

video_path = r"C:\Users\HKIT\project\.gitignore\26_08_10\drive-download-20260810T075024Z-1-001\DJI_20260702191744_0004_치맥페스티벌녹화영상.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(fps * 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

prev_time = 60  # FPS 계산용 변수

print("성능 비교 측정을 시작합니다. 'q'를 누르면 종료됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 시간 체크 (FPS 계산 시작)
    current_time = time.time()

    # 3. YOLO 객체 인식 수행 *추적코드 적용
    results = model.track(
    frame, conf=0.3, persist=True, imgsz=640, verbose=False
)

    # 4. 시각화
    annotated_frame = results[0].plot(font_size=0.3, line_width=1)

    # FPS 계산 로직
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # 화면에 실시간 FPS 텍스트 박아넣기 (병목 현상 수치화용)
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # 5. 화면 출력
    cv2.imshow("Performance Comparison Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("측정이 종료되었습니다.")