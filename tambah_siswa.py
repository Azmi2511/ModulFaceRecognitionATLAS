import cv2
import face_recognition
import os
import database as db

def register_new_student():
    """ Mendaftarkan siswa baru melalui webcam """
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    nis = input("Masukkan NIS Siswa: ")
    name = input("Masukkan Nama Lengkap Siswa: ")
    student_class = input("Masukkan Kelas Siswa: ")

    video_capture = cv2.VideoCapture(0)
    print("\nLihat ke kamera dan tekan 's' untuk menyimpan gambar. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Gagal mengakses kamera.")
            break
        
        cv2.imshow('Pendaftaran Wajah - Tekan s untuk simpan', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_path = os.path.join('dataset', f"{nis}_{name}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Gambar disimpan di {image_path}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) == 1:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                
                db.add_student(nis, name, student_class, face_encoding)
                break
            elif len(face_locations) > 1:
                print("Terdeteksi lebih dari satu wajah. Harap pastikan hanya ada satu wajah di depan kamera.")
            else:
                print("Wajah tidak terdeteksi. Coba lagi.")

        elif key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_new_student()