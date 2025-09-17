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

checked_in_today = db.get_checked_in_today()
print(f"Data kehadiran dimuat: {len(checked_in_today)} siswa sudah absen hari ini.")

last_attendance_check = 0
attendance_check_interval = 10

video_capture = None
camera_found = False

print("Mencari kamera yang tersedia...")
for camera_index in [0, 1, 2]:
    print(f"Mencoba kamera index {camera_index}...")
    video_capture = cv2.VideoCapture(camera_index)
    
    if video_capture.isOpened():
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

frame_width = 640
frame_height = 480
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
video_capture.set(cv2.CAP_PROP_FPS, 30)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

frame_skip = 2
frame_count = 0

last_face_detection_time = 0
face_detection_interval = 0.1

fps_start_time = time.time()
fps_frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera!")
        break
    
    if frame is None or frame.size == 0:
        print("Warning: Frame kosong, melanjutkan...")
        continue

    frame_count += 1
    fps_frame_count += 1

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    current_time = time.time()
    if frame_count % frame_skip == 0 and (current_time - last_face_detection_time) > face_detection_interval:
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog", number_of_times_to_upsample=1)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
        last_face_detection_time = current_time

        face_names = []
        for face_encoding in face_encodings:
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

                    if student_id not in checked_in_today:
                        current_time = time.time()
                        if current_time - last_attendance_check > attendance_check_interval:
                            if not db.is_already_checked_in(student_id):
                                if db.mark_attendance(student_id):
                                    checked_in_today.add(student_id)
                                    print(f"ABSENSI BERHASIL: {name} ({nis}) pada {datetime.now()}")
                            else:
                                checked_in_today.add(student_id)
                            last_attendance_check = current_time
                        else:
                            if not db.is_already_checked_in(student_id):
                                if db.mark_attendance(student_id):
                                    checked_in_today.add(student_id)
                                    print(f"ABSENSI BERHASIL: {name} ({nis}) pada {datetime.now()}")
                            else:
                                checked_in_today.add(student_id)
                    else:
                        name = f"{name} (Sudah Absen)"

            face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        color = (0, 255, 0) if "Sudah Absen" not in name else (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    if fps_frame_count % 30 == 0:
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            fps = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Cache: {len(checked_in_today)} siswa", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Video Absensi Wajah - Optimized', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        checked_in_today.clear()
        print("Cache kehadiran direset.")
    elif key == ord('c'):
        ret, test_frame = video_capture.read()
        if ret:
            print(f"Kamera OK - Frame size: {test_frame.shape}")
        else:
            print("Kamera ERROR - Tidak dapat membaca frame")

video_capture.release()
cv2.destroyAllWindows()