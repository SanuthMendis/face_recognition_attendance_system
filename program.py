import face_recognition
import numpy as np
import cv2
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

barakobama_image = face_recognition.load_image_file("photos/barakobama.jpg")
barakobama_encoding = face_recognition.face_encodings(barakobama_image)[0]

elonmusk_image = face_recognition.load_image_file("photos/elonmusk.jpg")
elonmusk_encoding = face_recognition.face_encodings(elonmusk_image)[0]

mahela_image = face_recognition.load_image_file("photos/mahela.jpg")
mahela_encoding = face_recognition.face_encodings(mahela_image)[0]

messi_image = face_recognition.load_image_file("photos/messi.jpg")
messi_encoding = face_recognition.face_encodings(messi_image)[0]

sanga_image = face_recognition.load_image_file("photos/sanga.jpg")
sanga_encoding = face_recognition.face_encodings(sanga_image)[0]

known_face_encoding = [
    barakobama_encoding,
    elonmusk_encoding,
    mahela_encoding,
    messi_encoding,
    sanga_encoding
]

known_face_names = [
    "Barak obama",
    "Elon Musk",
    "Mahela Jayawardhana",
    "Lionel Messi",
    "Kumar Sangakkara"

]

students = known_face_names.copy()

face_locations= []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("attendance_system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

