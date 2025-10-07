import cv2

for index in range(3):  # 測試索引 0, 1, 2
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera opened successfully at index {index}")
        ret, frame = cap.read()
        if ret:
            print(f"Frame captured at index {index}, shape: {frame.shape}")
            cv2.imshow(f'Camera {index}', frame)
            cv2.waitKey(1000)
        else:
            print(f"Failed to grab frame at index {index}")
        cap.release()
    else:
        print(f"Cannot open camera at index {index}")
cv2.destroyAllWindows()
