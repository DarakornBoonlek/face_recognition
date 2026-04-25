import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import my_app
import numpy as np
import os
import csv
from datetime import datetime

known_face_encodings = [] #สร้าง ลิสต์เปล่า สำหรับเก็บ เวกเตอร์ใบหน้า (face encodings) ของผู้ใช้ทั้งหมดที่รู้จัก
known_face_names = [] #สร้างลิสต์เปล่าอีกตัว เก็บ ชื่อผู้ใช้ ที่ตรงกับ encoding ในลิสต์ก่อนหน้า
known_folder = "known_faces" # กำหนดชื่อโฟลเดอร์ที่เก็บรูปภาพของใบหน้าที่รู้จัก → รูปทั้งหมดในโฟลเดอร์นี้จะถูกโหลดเข้าเป็นฐานข้อมูล
os.makedirs(known_folder, exist_ok=True) #ตรวจสอบว่าโฟลเดอร์ known_faces มีอยู่ไหม ถ้าไม่มี → สร้างโฟลเดอร์ให้ /ถ้ามีแล้ว → ไม่ error เพราะ exist_ok=True

# โหลดใบหน้าที่รู้จัก
for filename in os.listdir(known_folder): # วนลูปอ่าน ชื่อไฟล์ทั้งหมด ในโฟลเดอร์ known_faces
    if filename.endswith(('.jpg', '.png')): #รวจสอบว่าไฟล์นั้นเป็นภาพหรือไม่
        image_path = os.path.join(known_folder, filename) #สร้าง path สมบูรณ์ เช่น: "known_faces/john.jpg"
        image = my_app.load_image_file(image_path) #โหลดไฟล์ภาพเข้าเป็นข้อมูลแบบ numpy array
        encoding = my_app.face_encodings(image) # สร้าง เวกเตอร์ใบหน้า (128 ค่า) จากภาพ
        if encoding: #เช็คว่าเจอใบหน้าในภาพหรือไม่
            known_face_encodings.append(encoding[0]) #ถ้าเจอ → เพิ่ม encoding แรกเข้าในลิสต์ known_face_encodings
            name = os.path.splitext(filename)[0] # แยกชื่อไฟล์ออกจากนามสกุล เช่น "john.jpg" → "john" ดึงส่วนแรกของ tuple → 'john'
            known_face_names.append(name) # เพิ่มชื่อนั้นลงใน known_face_names

def log_user_entry(name): #สร้างฟังก์ชันชื่อ log_user_entry รับพารามิเตอร์ name = ชื่อของผู้ที่เข้าระบบ
    with open("logs.csv", mode='a', newline='') as file: #เปิดไฟล์ logs.csv ในโหมด append (เขียนต่อท้าย) ถ้าไฟล์ยังไม่มี → สร้างใหม่ / newline='' ป้องกันการเว้นบรรทัดเกินในบางระบบ
        writer = csv.writer(file) #สร้างตัวช่วยเขียนข้อมูลลงในไฟล์แบบ .csv
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") #ดึงวันเวลา ณ ปัจจุบัน แล้วแปลงให้อยู่ในรูปแบบ "2025-05-28 15:42:00"
        writer.writerow([name, now]) #เขียนแถวใหม่ในไฟล์ log โดยมี 2 คอลัมน์: ชื่อ, เวลาเข้าใช้งาน

def face_login(): #ประกาศฟังก์ชัน face_login() — เรียกใช้เมื่อผู้ใช้คลิกปุ่ม "ล็อกอินด้วยใบหน้า"
    cap = cv2.VideoCapture(0) #เปิดกล้อง WebCam (index 0 คือกล้องตัวหลัก)
    login_success = False #ตัวแปร flag เริ่มต้น = ยังไม่ล็อกอินสำเร็จ
    matched_name = "Unknown" #กำหนดชื่อเริ่มต้นว่า "ไม่รู้จัก" ก่อน → หากพบชื่อจริงจะถูกอัปเดต
    logged_names = set() #สร้างเซ็ตเปล่าไว้เก็บชื่อที่ ล็อกอินแล้วในรอบนี้ → ป้องกันบันทึกซ้ำ

    while True: #ลูปอ่านภาพจากกล้องแบบ real-time
        ret, frame = cap.read() #อ่านภาพจากกล้อง ret = ได้ภาพจริงหรือไม่ (True/False) frame = ภาพที่อ่านได้ (numpy array)
        if not ret: # ถ้าอ่านภาพไม่ได้ → ออกจากลูปทันที
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #แปลงภาพจาก BGR (OpenCV format) → RGB (ใช้กับ face_recognition)
        rgb_frame = np.ascontiguousarray(rgb_frame) #ทำให้ array จัดเรียงในหน่วยความจำแบบต่อเนื่อง → ป้องกัน error กับ dlib

        face_locations = my_app.face_locations(rgb_frame, model='hog') #ตรวจจับตำแหน่งใบหน้า (top, right, bottom, left) โดยใช้โมเดล hog
        face_encodings = my_app.face_encodings(rgb_frame, face_locations) #แปลงแต่ละใบหน้า → เป็นเวกเตอร์ (face encoding ขนาด 128 ค่า)
 
        for face_encoding in face_encodings: #วนลูปทีละใบหน้า (กรณีมีหลายคนในกล้อง)
            matches = my_app.compare_faces(known_face_encodings, face_encoding) #เปรียบเทียบใบหน้าที่เจอ กับฐานข้อมูล → ได้ list ของ True/False
            face_distances = my_app.face_distance(known_face_encodings, face_encoding) #วัดระยะห่าง (คล้ายความแม่นยำ) → ค่าใกล้ 0 = เหมือน

            if len(face_distances) > 0: # เช็กว่ามีข้อมูลในฐานหรือไม่ (ป้องกัน crash)
                best_match_index = np.argmin(face_distances) #หาค่า index ที่มีระยะห่างน้อยที่สุด (ใกล้ที่สุด)
                if matches[best_match_index]: # ถ้า index ที่ดีที่สุดตรงกับ True ใน matches → แสดงว่า “เจอชื่อ”
                    matched_name = known_face_names[best_match_index] #ดึงชื่อจาก index ที่เจอ → เช่น "john"
                    if matched_name not in logged_names: #ถ้าชื่อนั้นยังไม่เคย log มาก่อนในรอบนี้
                        log_user_entry(matched_name) # เรียกฟังก์ชัน log_user_entry() เพื่อบันทึกชื่อ + เวลา ลง logs.csv
                        logged_names.add(matched_name) # เพิ่มชื่อเข้าในเซ็ต → เพื่อไม่ให้บันทึกซ้ำอีก
                    login_success = True #ตั้งสถานะว่า "ล็อกอินสำเร็จ" → ใช้ตัดลูป
                    break #ออกจาก for loop → ไม่ต้องตรวจใบหน้าอื่นแล้ว

        cv2.imshow("Face Login", frame) #แสดงภาพจากกล้องพร้อมกรอบในหน้าต่างชื่อ "Face Login"
        if login_success or cv2.waitKey(1) & 0xFF == ord('q'): #ถ้าล็อกอินสำเร็จ หรือกดปุ่ม q → หยุดลูป
            break 

    cap.release() #ปิดการใช้งานกล้อง 
    cv2.destroyAllWindows() #ปิดหน้าต่างภาพทั้งหมดของ OpenCV

    if login_success: #เช็กว่าเราล็อกอินสำเร็จหรือไม่
        messagebox.showinfo("ล็อกอินสำเร็จ", f"ยินดีต้อนรับคุณ {matched_name}") # แสดง popup ว่า “ล็อกอินสำเร็จ” พร้อมชื่อ
    else:
        messagebox.showwarning("ไม่รู้จักใบหน้า", "❌ ไม่สามารถระบุผู้ใช้งานได้") #ถ้าไม่เจอชื่อ → แจ้งเตือนว่า "ไม่รู้จักใบหน้า"

def register_user(): # สร้างฟังก์ชันชื่อ register_user()
    name = simpledialog.askstring("ลงทะเบียน", "กรุณากรอกชื่อผู้ใช้ (อังกฤษเท่านั้น):") # เปิดกล่องให้ผู้ใช้พิมพ์ชื่อ (ภาษาอังกฤษ) → ใช้ชื่อที่กรอกเป็นชื่อไฟล์รูปภาพใน known_faces/
    if not name: #ถ้าผู้ใช้ไม่กรอก หรือกด "Cancel" → หยุดฟังก์ชันทันที
        return

    cap = cv2.VideoCapture(0) #เปิดกล้องของเครื่อง (WebCam)
    while True: # เริ่มลูปแสดงภาพจากกล้องแบบ real-time
        ret, frame = cap.read() # อ่านภาพจากกล้อง ret = อ่านสำเร็จหรือไม่ / frame = ภาพที่ได้
        if not ret: #ถ้าอ่านภาพไม่สำเร็จ → ออกจากลูป
            break

        cv2.imshow("ลงทะเบียน - กด 's' เพื่อบันทึก", frame) #แสดงภาพจากกล้องพร้อมข้อความบอกว่าให้กด 's' เพื่อบันทึกใบหน้า
        key = cv2.waitKey(1) #รอการกดคีย์บอร์ด 1 มิลลิวินาที → รับค่าที่กดเก็บไว้ใน key
        if key & 0xFF == ord('s'): #ถ้ากดปุ่ม 's' → เข้าสู่ขั้นตอนบันทึกใบหน้า
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #แปลงภาพจาก BGR (แบบ OpenCV) เป็น RGB (แบบ face_recognition ใช้)
            faces = my_app.face_locations(rgb) #ตรวจจับตำแหน่งใบหน้าในภาพ → ได้ list ของ (top, right, bottom, left)
            if faces: #ถ้าเจอใบหน้าอย่างน้อย 1 ใบ → ดำเนินการต่อ
                top, right, bottom, left = faces[0] #ดึงตำแหน่งใบหน้าใบแรกออกมา (เผื่อมีหลายใบหน้าในภาพ)
                face_image = frame[top:bottom, left:right] #ตัดเฉพาะส่วนที่เป็นใบหน้าออกจากภาพ frame (ใช้กับ BGR เพราะจะเซฟเป็นรูป)
                path = os.path.join(known_folder, f"{name}.jpg") # สร้าง path เต็มของไฟล์ที่จะบันทึก เช่น:known_faces/alex.jpg
                cv2.imwrite(path, face_image) #บันทึกภาพใบหน้าที่ตัดมาเป็น .jpg ในโฟลเดอร์ known_faces/
                messagebox.showinfo("สำเร็จ", f"✅ บันทึกใบหน้าของ {name} แล้ว") #แสดงกล่องข้อความแจ้งว่า “บันทึกใบหน้าเรียบร้อยแล้ว”
            else:
                messagebox.showwarning("ไม่พบใบหน้า", "⚠️ กรุณาหันหน้าให้ชัดเจน") #ถ้าไม่เจอใบหน้า → แจ้งเตือนให้หันหน้าให้ชัดเจน แล้วลองใหม่


            break #ออกจากลูปลงทะเบียนทันที (หลังบันทึกหรือแจ้งเตือนแล้ว)
        elif key & 0xFF == ord('q'): #ถ้ากดปุ่ม 'q' → ยกเลิกการลงทะเบียน → ออกจากลูป
            break

    cap.release() #ปิดกล้อง
    cv2.destroyAllWindows() #ปิดหน้าต่างกล้องทั้งหมด

def show_logs(): #สร้างฟังก์ชันชื่อ show_logs() สำหรับเปิดดู log การเข้าใช้งาน (จาก logs.csv)
    if not os.path.exists("logs.csv"): #ถ้ายังไม่มีไฟล์ logs.csv → แสดงว่าไม่มีข้อมูลการเข้าใช้งาน
        messagebox.showinfo("Logs", "ยังไม่มีการบันทึกการเข้าใช้งาน") #แสดง popup แจ้งผู้ใช้ว่า "ยังไม่มีการบันทึก" แล้ว ออกจากฟังก์ชันทันที
        return

    with open("logs.csv", newline='') as file: # เปิดไฟล์ logs.csv แบบอ่าน (read) พร้อมกำหนด newline='' ป้องกันเว้นบรรทัดเกิน
        reader = csv.reader(file) #อ่านข้อมูลในไฟล์แบบแถวละรายการด้วย csv.reader
        logs = "\n".join([f"{name} - {time}" for name, time in reader]) #รวมทุกแถวเป็นข้อความหลายบรรทัด เช่น john - 2025-05-28 14:00:01  
    messagebox.showinfo("Log การเข้าใช้งาน", logs or "ไม่มีข้อมูล") #แสดง popup พร้อมข้อความทั้งหมดที่อ่านได้

# สร้าง GUI หลัก
root = tk.Tk() #สร้างหน้าต่างหลัก (Tk root window) สำหรับ GUI
root.title("ระบบยืนยันตัวตนด้วยใบหน้า") #ตั้งชื่อของหน้าต่างโปรแกรม
root.geometry("400x300") #กำหนดขนาดหน้าต่างกว้าง 400 สูง 300 พิกเซล
root.resizable(False, False) # ไม่อนุญาตให้ผู้ใช้ปรับขนาดหน้าต่าง (แนวนอน/แนวตั้ง)

tk.Label(root, text="Face Authentication", font=("Arial", 18)).pack(pady=20) #สร้างข้อความหัวเรื่องด้วยฟอนต์ขนาดใหญ่ → ใช้ .pack() เพื่อวางบนหน้าต่างพร้อมระยะห่างแนวตั้ง 20 px
 
tk.Button(root, text="🔓 ล็อกอินด้วยใบหน้า", font=("Arial", 14), command=face_login).pack(pady=10) #สร้างปุ่ม "ล็อกอินด้วยใบหน้า" → เมื่อกดจะเรียก face_login() ฟอนต์ขนาด 14, icon ช่วยให้ดูเข้าใจง่ายขึ้น
tk.Button(root, text="➕ ลงทะเบียนผู้ใช้ใหม่", font=("Arial", 14), command=register_user).pack(pady=10) #ปุ่ม "ลงทะเบียนผู้ใช้ใหม่" → เมื่อกดจะเรียก register_user()
tk.Button(root, text="📄 แสดงประวัติการเข้าใช้งาน", font=("Arial", 12), command=show_logs).pack(pady=10) #ปุ่ม “แสดง log” → เรียก show_logs() → เปิด popup แสดง log ที่อยู่ใน logs.csv
tk.Button(root, text="❌ ออก", font=("Arial", 12), command=root.quit).pack(pady=20) #ปุ่ม "ออก" → เมื่อกดจะปิดแอปทันทีด้วย root.quit

root.mainloop() # เริ่มการทำงานของ GUI (รอรับเหตุการณ์จากผู้ใช้ เช่น การกดปุ่ม)
