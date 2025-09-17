import cv2
import face_recognition
import numpy as np
import database as db
from datetime import datetime
import time
import threading

print("Memuat data wajah dari database...")
known_face_encodings, known_face_metadata = db.get_known_faces()
print("Data wajah berhasil dimuat.")

# Cache untuk siswa yang sudah absen hari ini
print("Memuat data kehadiran hari ini...")
checked_in_today = db.get_checked_in_today()
print(f"Data kehadiran dimuat: {len(checked_in_today)} siswa sudah absen hari ini.")

last_attendance_check = 0
attendance_check_interval = 10  # Cek database setiap 10 detik

# Coba kamera dengan index yang berbeda
video_capture = None
camera_found = False

print("Mencari kamera yang tersedia...")
for camera_index in [0, 1, 2]:  # Mulai dari index 0 karena test menunjukkan index 0 berfungsi
    print(f"Mencoba kamera index {camera_index}...")
    video_capture = cv2.VideoCapture(camera_index)
    
    if video_capture.isOpened():
        # Test apakah kamera benar-benar bisa membaca frame
        ret, test_frame = video_capture.read()
        if ret and test_frame is not None and test_frame.size > 0:
            print(f"✓ Kamera {camera_index} berfungsi - Resolusi: {test_frame.shape[1]}x{test_frame.shape[0]}")
            camera_found = True
            break
        else:
            print(f"✗ Kamera {camera_index} tidak dapat membaca frame")
            video_capture.release()
    else:
        print(f"✗ Kamera {camera_index} tidak dapat dibuka")
        video_capture.release()

if not camera_found or video_capture is None or not video_capture.isOpened():
    print("\nError: Tidak dapat mengakses kamera!")
    print("Solusi:")
    print("1. Jalankan test_camera.py untuk mencari kamera yang berfungsi")
    print("2. Pastikan kamera tidak digunakan aplikasi lain")
    print("3. Cek driver kamera")
    input("Tekan Enter untuk keluar...")
    exit()

# Optimasi: kurangi ukuran frame untuk deteksi wajah
frame_width = 640
frame_height = 480
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Optimasi tambahan untuk performa
video_capture.set(cv2.CAP_PROP_FPS, 30)  # Set FPS ke 30
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Kurangi buffer untuk latency rendah

# Kamera sudah ditest sebelumnya, langsung lanjut

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Optimasi: proses setiap 2 frame untuk performa yang lebih baik
frame_skip = 2
frame_count = 0

# Variabel untuk optimasi performa
last_face_detection_time = 0
face_detection_interval = 0.1  # Deteksi wajah setiap 100ms

# Variabel untuk FPS
fps_start_time = time.time()
fps_frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera!")
        break
    
    # Pastikan frame tidak kosong
    if frame is None or frame.size == 0:
        print("Warning: Frame kosong, melanjutkan...")
        continue

    frame_count += 1
    fps_frame_count += 1

    # Optimasi: resize frame lebih kecil untuk deteksi wajah
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Ubah dari 0.25 ke 0.5 untuk kualitas lebih baik
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Optimasi: proses hanya setiap frame_skip frame dan interval waktu
    current_time = time.time()
    if frame_count % frame_skip == 0 and (current_time - last_face_detection_time) > face_detection_interval:
        # Gunakan deteksi wajah yang lebih cepat
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog", number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)  # Kurangi jitters untuk performa
        last_face_detection_time = current_time

        face_names = []
        for face_encoding in face_encodings:
            # Optimasi: gunakan tolerance yang lebih ketat untuk performa
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Tidak Dikenal"
            
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    metadata = known_face_metadata[best_match_index]
                    student_id = metadata["id"]
                    nis = metadata["nis"]
                    name = metadata["name"]

                    # Optimasi: cek cache dulu sebelum database
                    if student_id not in checked_in_today:
                        # Cek database hanya jika belum ada di cache
                        current_time = time.time()
                        if current_time - last_attendance_check > attendance_check_interval:
                            # Update cache dari database secara batch
                            if not db.is_already_checked_in(student_id):
                                if db.mark_attendance(student_id):
                                    checked_in_today.add(student_id)
                                    print(f"ABSENSI BERHASIL: {name} ({nis}) pada {datetime.now()}")
                            else:
                                checked_in_today.add(student_id)
                            last_attendance_check = current_time
                        else:
                            # Gunakan cache untuk performa
                            if not db.is_already_checked_in(student_id):
                                if db.mark_attendance(student_id):
                                    checked_in_today.add(student_id)
                                    print(f"ABSENSI BERHASIL: {name} ({nis}) pada {datetime.now()}")
                            else:
                                checked_in_today.add(student_id)
                    else:
                        name = f"{name} (Sudah Absen)"

            face_names.append(name)

    # Tampilkan kotak dan nama wajah
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2  # Ubah dari 4 ke 2 karena resize 0.5x
        right *= 2
        bottom *= 2
        left *= 2

        # Warna berbeda untuk status
        color = (0, 255, 0) if "Sudah Absen" not in name else (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Hitung dan tampilkan FPS (hanya setiap 30 frame untuk performa)
    if fps_frame_count % 30 == 0:
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:  # Update FPS setiap detik
            fps = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
            
            # Tampilkan FPS di frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Cache: {len(checked_in_today)} siswa", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tampilkan frame
    cv2.imshow('Video Absensi Wajah - Optimized', frame)

    # Optimasi: kurangi delay untuk responsivitas yang lebih baik
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset cache dengan tombol 'r'
        checked_in_today.clear()
        print("Cache kehadiran direset.")
    elif key == ord('c'):  # Test kamera dengan tombol 'c'
        ret, test_frame = video_capture.read()
        if ret:
            print(f"Kamera OK - Frame size: {test_frame.shape}")
        else:
            print("Kamera ERROR - Tidak dapat membaca frame")

video_capture.release()
cv2.destroyAllWindows()