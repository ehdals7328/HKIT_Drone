import cv2
import time
from ultralytics import YOLO

# 1. Yolo26n 모델에 VisDrone 데이터 셋을 학습시킨 pt임
model = YOLO('best_.pt')

video_path = r"C:\Users\HKIT\project\.gitignore\26_08_10\drive-download-20260810T075024Z-1-003\GCS녹화_2024_12_04_11_57_울주군도로노면촬영.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

prev_time = time.time() #FPS 계산용 변수

print("성능 비교 측정을 시작합니다. 'q'를 누르면 종료됩니다.")

while cap.isOpened():
    ret, frame = cap.read() # frame은 1920x1080 원본 전체
    if not ret:
        break
    
    detect_frame = frame.copy() # 3번 영상을 위한 위아래 마스킹 (UI 오인식 방지)
    detect_frame[0:85, :] = 0
    detect_frame[975:, :] = 0
    
    results = model.track(detect_frame, conf=0.3, persist=True, imgsz=640, verbose=False) # 위아래가 지워진 영상을 최종 전달
    annotated_frame = results[0].plot(img=frame, font_size=0.3, line_width=1) # 라벨링 박스 UI

    current_time = time.time() # fps 표현 (좌상단)
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    
    cv2.imshow("Drone Detection (Full Screen)", annotated_frame) #최종 출력
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("측정이 종료되었습니다.")