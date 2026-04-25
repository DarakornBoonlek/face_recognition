import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

known_face_encodings = [] 
known_face_names = [] 
known_folder = "known_faces" 
os.makedirs(known_folder, exist_ok=True) 

# โหลดใบหน้าที่รู้จัก
for filename in os.listdir(known_folder): 
    if filename.endswith(('.jpg', '.png')): 
        image_path = os.path.join(known_folder, filename) 
        image = face_recognition.load_image_file(image_path) 
        encoding = face_recognition.face_encodings(image) 
        if encoding: 
            known_face_encodings.append(encoding[0]) 
            name = os.path.splitext(filename)[0] 
            known_face_names.append(name) 

def log_user_entry(name): 
    with open("logs.csv", mode='a', newline='') as file: 
        writer = csv.writer(file) 
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        writer.writerow([name, now]) 

def face_login(): 
    cap = cv2.VideoCapture(0) 
    login_success = False 
    matched_name = "Unknown" 
    logged_names = set() 

    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        rgb_frame = np.ascontiguousarray(rgb_frame) 

        face_locations = face_recognition.face_locations(rgb_frame, model='hog') 
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations) 
 
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings): 
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding) 
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding) 
            matched_name = "Unknown"
            
            if len(face_distances) > 0: 
                best_match_index = np.argmin(face_distances) 
                if matches[best_match_index]: 
                    matched_name = known_face_names[best_match_index] 
                    if matched_name not in logged_names: 
                        log_user_entry(matched_name) 
                        logged_names.add(matched_name) 
                    login_success = True 
                    break # เมื่อล็อกอินสำเร็จ ให้ออกจาก for loop

            # วาดกรอบสี่เหลี่ยมรอบใบหน้าในหน้าจอ Login (ไม่ใช้ Padding เพราะแค่วาดโชว์)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, matched_name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Face Login", frame) 
        if login_success or cv2.waitKey(1) & 0xFF == ord('q'): 
            break 

    cap.release() 
    cv2.destroyAllWindows() 

    if login_success: 
        messagebox.showinfo("ล็อกอินสำเร็จ", f"ยินดีต้อนรับคุณ {matched_name}") 
    else:
        messagebox.showwarning("ไม่รู้จักใบหน้า", "❌ ไม่สามารถระบุผู้ใช้งานได้") 

def register_user(): 
    name = simpledialog.askstring("ลงทะเบียน", "กรุณากรอกชื่อผู้ใช้ (อังกฤษเท่านั้น):") 
    if not name: 
        return

    cap = cv2.VideoCapture(0) 
    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break

        cv2.imshow("Register - Press 's' to save", frame) 
        key = cv2.waitKey(1) 
        if key & 0xFF == ord('s'): 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            faces = face_recognition.face_locations(rgb) 
            
            if faces: 
                top, right, bottom, left = faces[0] 
                
                # --- คำนวณ Padding ก่อนตัดรูปภาพเพื่อเซฟ ---
                face_h = bottom - top
                padding = int(0.25 * face_h)
                top_pad = max(0, top - padding)
                left_pad = max(0, left - padding)
                bottom_pad = min(frame.shape[0], bottom + padding)
                right_pad = min(frame.shape[1], right + padding)
                
                # ตัดภาพใบหน้าโดยใช้ระยะขอบที่คำนวณแล้ว
                face_image = frame[top_pad:bottom_pad, left_pad:right_pad] 
                
                path = os.path.join(known_folder, f"{name}.jpg") 
                cv2.imwrite(path, face_image) 
                
                # นำชื่อและใบหน้าใหม่ เข้าไปในหน่วยความจำทันที (จะได้ไม่ต้องปิดเปิดโปรแกรมใหม่)
                new_image = face_recognition.load_image_file(path)
                new_encoding = face_recognition.face_encodings(new_image)
                if new_encoding:
                    known_face_encodings.append(new_encoding[0])
                    known_face_names.append(name)

                messagebox.showinfo("สำเร็จ", f"✅ บันทึกใบหน้าของ {name} แล้ว") 
            else:
                messagebox.showwarning("ไม่พบใบหน้า", "⚠️ กรุณาหันหน้าให้ชัดเจน") 

            break 
        elif key & 0xFF == ord('q'): 
            break

    cap.release() 
    cv2.destroyAllWindows() 

def show_logs(): 
    if not os.path.exists("logs.csv"): 
        messagebox.showinfo("Logs", "ยังไม่มีการบันทึกการเข้าใช้งาน") 
        return

    with open("logs.csv", newline='') as file: 
        reader = csv.reader(file) 
        logs = "\n".join([f"{name} - {time}" for name, time in reader])  
    messagebox.showinfo("Log การเข้าใช้งาน", logs or "ไม่มีข้อมูล") 

# สร้าง GUI หลัก
root = tk.Tk() 
root.title("ระบบยืนยันตัวตนด้วยใบหน้า") 
root.geometry("400x300") 
root.resizable(False, False) 

tk.Label(root, text="Face Authentication", font=("Arial", 18)).pack(pady=20) 
tk.Button(root, text="🔓 ล็อกอินด้วยใบหน้า", font=("Arial", 14), command=face_login).pack(pady=10) 
tk.Button(root, text="➕ ลงทะเบียนผู้ใช้ใหม่", font=("Arial", 14), command=register_user).pack(pady=10) 
tk.Button(root, text="📄 แสดงประวัติการเข้าใช้งาน", font=("Arial", 12), command=show_logs).pack(pady=10) 
tk.Button(root, text="❌ ออก", font=("Arial", 12), command=root.quit).pack(pady=20) 

root.mainloop()