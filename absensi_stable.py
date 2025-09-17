import cv2
import face_recognition
import numpy as np
import database as db
from datetime import datetime
import time

print("Memuat data wajah dari database...")
known_face_encodings, known_face_metadata = db.get_known_faces()
print("Data wajah berhasil dimuat.")

# Cache untuk siswa yang sudah absen hari ini
print("Memuat data kehadiran hari ini...")
checked_in_today = db.get_checked_in_today()
print(f"Data kehadiran dimuat: {len(checked_in_today)} siswa sudah absen hari ini.")

# Inisialisasi kamera dengan backend DSHOW (lebih stabil di Windows)
print("Menginisialisasi kamera...")
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print("Mencoba kamera index 1...")
    video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print("Mencoba kamera index 2...")
    video_capture = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print("Error: Tidak dapat mengakses kamera!")
    input("Tekan Enter untuk keluar...")
    exit()

# Test kamera
ret, test_frame = video_capture.read()
if not ret or test_frame is None:
    print("Error: Kamera tidak dapat membaca frame!")
    video_capture.release()
    input("Tekan Enter untuk keluar...")
    exit()

print(f"✓ Kamera berfungsi - Resolusi: {test_frame.shape[1]}x{test_frame.shape[0]}")

# Set properti kamera
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Variabel untuk optimasi
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
last_detection_time = 0
detection_interval = 0.3  # Deteksi setiap 300ms untuk stabilitas

print("\nSistem absensi dimulai!")
print("Tekan 'q' untuk keluar, 'r' untuk reset cache")

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera!")
            break
        
        if frame is None or frame.size == 0:
            continue

        frame_count += 1
        current_time = time.time()
        
        # Deteksi wajah hanya setiap interval tertentu
        if current_time - last_detection_time > detection_interval:
            try:
                # Resize frame untuk deteksi
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Deteksi wajah dengan error handling
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
                
                face_names = []
                for face_encoding in face_encodings:
                    try:
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
                                    if not db.is_already_checked_in(student_id):
                                        if db.mark_attendance(student_id):
                                            checked_in_today.add(student_id)
                                            print(f"ABSENSI BERHASIL: {name} ({nis}) pada {datetime.now()}")
                                    else:
                                        checked_in_today.add(student_id)
                                else:
                                    name = f"{name} (Sudah Absen)"

                        face_names.append(name)
                    except Exception as e:
                        print(f"Error dalam face recognition: {e}")
                        face_names.append("Error")
                
                last_detection_time = current_time
            except Exception as e:
                print(f"Error dalam deteksi wajah: {e}")
                face_locations = []
                face_encodings = []
                face_names = []

        # Tampilkan kotak dan nama wajah
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

        # Tampilkan informasi
        cv2.putText(frame, f"Cache: {len(checked_in_today)} siswa", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Tekan 'q' keluar, 'r' reset", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Video Absensi Wajah - Stable', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            checked_in_today.clear()
            print("Cache kehadiran direset.")

except KeyboardInterrupt:
    print("\nProgram dihentikan oleh user")
except Exception as e:
    print(f"Error: {e}")
finally:
    video_capture.release()
    cv2.destroyAllWindows()
    print("Sistem absensi dihentikan.")
