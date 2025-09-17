import mysql.connector
from configparser import ConfigParser
import numpy as np
import pickle
from datetime import date

def read_db_config(filename='config.ini', section='database'):
    """ Membaca konfigurasi database dari file """
    parser = ConfigParser()
    parser.read(filename)
    db_config = {}
    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            db_config[item[0]] = item[1]
    else:
        raise Exception(f'{section} not found in the {filename} file')
    return db_config

def get_connection():
    """ Membuat koneksi ke database """
    try:
        config = read_db_config()
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def add_student(nis, name, student_class, face_encoding):
    """ Menambahkan siswa baru ke database """
    conn = get_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        pickled_encoding = pickle.dumps(face_encoding)
        
        query = "INSERT INTO students (nis, name, class, face_encoding) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (nis, name, student_class, pickled_encoding))
        conn.commit()
        print(f"Siswa {name} dengan NIS {nis} berhasil ditambahkan.")
    except mysql.connector.Error as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

def get_known_faces():
    """ Mengambil semua data wajah yang sudah terdaftar dari database """
    conn = get_connection()
    if conn is None:
        return [], []

    known_face_encodings = []
    known_face_metadata = [] 
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, nis, name, face_encoding FROM students")
        records = cursor.fetchall()
        
        for row in records:
            student_id, nis, name, pickled_encoding = row
            encoding = pickle.loads(pickled_encoding)
            known_face_encodings.append(encoding)
            known_face_metadata.append({
                "id": student_id,
                "nis": nis,
                "name": name
            })
            
    except mysql.connector.Error as e:
        print(f"Error fetching data: {e}")
    finally:
        cursor.close()
        conn.close()
        
    return known_face_encodings, known_face_metadata

def is_already_checked_in(student_id):
    """ Cek apakah siswa sudah absen hari ini """
    conn = get_connection()
    if conn is None:
        return True 

    already_checked_in = False
    try:
        cursor = conn.cursor()
        today = date.today()
        # Optimasi: gunakan LIMIT 1 untuk performa yang lebih baik
        query = "SELECT id FROM attendance WHERE student_id = %s AND DATE(timestamp) = %s LIMIT 1"
        cursor.execute(query, (student_id, today))
        if cursor.fetchone():
            already_checked_in = True
    except mysql.connector.Error as e:
        print(f"Error checking attendance: {e}")
    finally:
        cursor.close()
        conn.close()
    return already_checked_in

def get_checked_in_today():
    """ Mendapatkan semua siswa yang sudah absen hari ini """
    conn = get_connection()
    if conn is None:
        return set()

    checked_in = set()
    try:
        cursor = conn.cursor()
        today = date.today()
        query = "SELECT DISTINCT student_id FROM attendance WHERE DATE(timestamp) = %s"
        cursor.execute(query, (today,))
        records = cursor.fetchall()
        
        for row in records:
            checked_in.add(row[0])
            
    except mysql.connector.Error as e:
        print(f"Error fetching checked in students: {e}")
    finally:
        cursor.close()
        conn.close()
        
    return checked_in

def mark_attendance(student_id):
    """ Mencatat kehadiran siswa ke tabel attendance """
    if is_already_checked_in(student_id):
        print(f"Siswa dengan ID {student_id} sudah absen hari ini.")
        return False
        
    conn = get_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        query = "INSERT INTO attendance (student_id, timestamp) VALUES (%s, NOW())"
        cursor.execute(query, (student_id,))
        conn.commit()
        print(f"Kehadiran untuk siswa ID {student_id} berhasil dicatat.")
        return True
    except mysql.connector.Error as e:
        print(f"Error marking attendance: {e}")
        return False
    finally:
        cursor.close()
        conn.close()