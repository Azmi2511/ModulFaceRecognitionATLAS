# Modul Pengenalan Wajah Project ATLAS

Modul ini menggunakan teknologi pengenalan wajah untuk mencatat kehadiran siswa secara otomatis. Data siswa dan absensi tersimpan di database MySQL.

## Fitur

- Absensi otomatis menggunakan kamera dan pengenalan wajah
- Penambahan data siswa baru beserta data wajah
- Penyimpanan data absensi dan siswa di database MySQL

## Instalasi

Pastikan Python 3.7+ sudah terinstal di komputer Anda.

Instal semua library yang dibutuhkan dengan perintah berikut:

```sh
pip install opencv-python face_recognition numpy mysql-connector-python
```

## Konfigurasi Database

1. Buat database dan tabel dengan query berikut di MySQL:

    ```sql
    CREATE DATABASE IF NOT EXISTS atlas_absensi;
    USE atlas_absensi;

    CREATE TABLE IF NOT EXISTS siswa (
        id INT AUTO_INCREMENT PRIMARY KEY,
        nama VARCHAR(100) NOT NULL,
        nis VARCHAR(20) NOT NULL UNIQUE,
        kelas VARCHAR(20) NOT NULL,
        wajah LONGBLOB
    );

    CREATE TABLE IF NOT EXISTS absensi (
        id INT AUTO_INCREMENT PRIMARY KEY,
        siswa_id INT NOT NULL,
        waktu DATETIME NOT NULL,
        status VARCHAR(20) NOT NULL,
        FOREIGN KEY (siswa_id) REFERENCES siswa(id)
    );
    ```

2. Edit file `config.ini` sesuai dengan pengaturan MySQL Anda (host, user, password, database).

## Cara Penggunaan

- **Absensi:** Jalankan salah satu file berikut sesuai kebutuhan:
  - `absensi.py`
  - `absensi_simple.py`
  - `absensi_stable.py`
  - `absensi_threaded.py`
- **Tambah Siswa:** Jalankan `tambah_siswa.py` untuk menambah data siswa baru beserta data wajah.

## Kontribusi

Kontribusi sangat terbuka! Silakan buat pull request atau buka issue untuk perbaikan dan pengembangan fitur.

## Lisensi

Proyek ini menggunakan lisensi MIT. Silakan gunakan dan modifikasi sesuai kebutuhan.