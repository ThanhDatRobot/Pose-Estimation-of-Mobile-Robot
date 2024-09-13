import cv2
import time
import os
from datetime import datetime

def capture_image():
    # Tạo thư mục lưu ảnh nếu chưa tồn tại
    save_folder = "captured_images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Khởi tạo webcam (nếu bạn sử dụng camera khác, thay đổi URL tương ứng)
    cap = cv2.VideoCapture('http://192.168.100.6:8080/video')

    # Kiểm tra xem có mở được camera hay không
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    try:
        # Đọc khung hình từ camera
        ret, frame = cap.read()

        if not ret:
            print("Không thể nhận khung hình. Kết thúc...")
            return

        # Hiển thị khung hình trong cửa sổ
        cv2.imshow('Captured Image', frame)

        # Lưu khung hình vào tệp (đặt tên tệp dựa trên thời gian hiện tại đến mili giây)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{save_folder}/captured_image_{current_time}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Đã lưu ảnh thành công: {filename}")

        # Chờ một chút để hiển thị ảnh
        cv2.waitKey(1000)

    finally:
        # Giải phóng tài nguyên và đóng cửa sổ
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
