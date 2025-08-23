import cv2

# RTSP 地址
rtsp_url = "rtsp://77.110.228.219/axis-media/media.amp"

# 打开 RTSP 视频流
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("无法打开视频流")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        break

    # 显示视频
    cv2.imshow("RTSP Video", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
