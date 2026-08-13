import cv2
from ultralytics import YOLO

# 1. 가장 가벼운 YOLO Nano 모델 로드 (처음 실행 시 자동으로 다운로드됩니다)
model = YOLO('best.pt')

# 2. 영상 경로 설정
# - 컴퓨터에 있는 동영상 파일로 테스트하려면: 'test_video.mp4' (파일 경로를 맞추어 주세요)
# - PC 웹캠이나 노트북 카메라도 테스트하고 싶다면: 0 으로 설정하세요.
video_path = r"C:\Users\samuel\Desktop\object_detection\DJI_20260702191744_0004_치맥페스티벌녹화영상.MP4"  # 웹캠으로 테스트하려면 0, 동영상 파일이라면 '동영상파일이름.mp4' 입력
#video_path = r"C:\Users\samuel\Desktop\object_detection\DJI_20260702182722_0004_치맥페스티벌녹화영상.MP4"  # 웹캠으로 테스트하려면 0, 동영상 파일이라면 '동영상파일이름.mp4' 입력
#video_path = r"C:\Users\samuel\Desktop\object_detection\GCS녹화_2024_12_04_11_57_울주군도로노면촬영.mp4"  # 웹캠으로 테스트하려면 0, 동영상 파일이라면 '동영상파일이름.mp4' 입력



cap = cv2.VideoCapture(video_path)
#비디오 mp4 영상을 cv타입으로 바꿔서 변수에 저장
if not cap.isOpened():
    print("영상을 열 수 없습니다. 파일 경로 또는 카메라 연결을 확인해주세요.")
    exit()

print("객체 인식을 시작합니다. 종료하려면 실행된 영상 창을 클릭하고 키보드 'q'를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read() #c타입 영상 객체를 행렬 변수 두개로 나누어 저장
    if not ret:
        print("영상이 끝났거나 더 이상 프레임을 읽을 수 없습니다.")
        break

    # 3. YOLO로 객체 인식 수행
    results = model(frame) # (frame, conf=0.5)

    # 4. 인식된 박스와 라벨을 프레임에 시각화
    annotated_frame = results[0].plot(font_size=0.3, line_width=1)

    # 5. 화면에 결과 출력
    cv2.imshow("YOLOv8 Video Object Detection", annotated_frame)

    # 키보드 'q'를 누르면 반복문 탈출 (창이 켜진 상태에서 키를 눌러야 합니다)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 사용이 끝난 자원 해제
cap.release()
cv2.destroyAllWindows()
print("프로그램이 종료되었습니다.")