import tempfile
import streamlit as st
from PIL import Image
import cv2
import csv
import os
import boto3
import moviepy.editor as moviepy
import collections
import numpy as np
from botocore.exceptions import ClientError
from matplotlib import pyplot as plt

from tracker import *

# Tracker
tracker = Trackers()


input_size = 255


confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2


middle_line_position = 255
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


#  Coco Names
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# indice de classe pour nos classes de détection requises
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Modeles
modelConfiguration = 'C:/Users/bowet/Downloads/vehicle-detection/models/yolov4.cfg'
modelWeigheights = 'C:/Users/bowet/Downloads/vehicle-detection/models/yolov4.weights'


net = cv2.dnn.readNet(modelConfiguration, modelWeigheights)


net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')



#
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    

temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]


def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id


    center = find_center(x, y, w, h)
    ix, iy = center
    
    # position du vehicule
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1


    cv2.circle(img, center, 2, (0, 0, 255), -1)
    # print(up_list, down_list)


# Fonction permettant de retrouver les objets détectés à partir du network output
def Processing(outputs,img):

    global detected_classNames 
    height, width = img.shape[:2]



    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))



    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    items = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            # recuperer le label de chaque objet
            label = str.upper((classNames[classIds[i]]))

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Dessinez le nom de la classe et le score de confiance
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Tracer le rectangle de délimitation
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            items.append(label)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    #Mettre à jour le tracker pour chaque objet

    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime(img, cap):

    while True:
        success, img = cap.read()

        try:
            img = cv2.resize(img,(0,0),None,0.5,0.5)
            # print(img.shape, "shape")
        except:
            break
        ih, iw, channels = img.shape

        fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        out = cv2.VideoWriter("data/ouput/detected_video.mp4", fourcc, 20.0, (iw, ih))



        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)


        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        # Alimenter network
        outputs = net.forward(outputNames)
    
        # Find the objects from the network output
        Processing(outputs, img)

        # Tracez les lignes de croisement

        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

        # Dessinez des textes de comptage dans le cadre
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)


        out.write(img)


        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    # sauvegarder le nombre de vehicule detecté

    with open("data/ouput/data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    print("Data saved at 'data.csv'")

    # Sauvegarde output sur s3

    access_key = 'AKIASN67I6KKE2L4ZTVI'
    secret_key = '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'
    bucket_name = 'projeta'

    # Connexion s3

    client_s3 = boto3.client(
        service_name='s3',
        region_name="eu-west-3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    # bucket = client_s3.Bucket(bucket_name)

    # transfer du fichier dans le compartiment s3

    data_folder = r'C:/Users/bowet/Downloads/vehicle-detection/data/ouput/'

    for file in os.listdir(data_folder):
        if not file.startswith('~'):
            try:
                print("uploading")
                client_s3.upload_file(
                    os.path.join(data_folder, file),
                    bucket_name,
                    file
                )
            except ClientError as e:
                print('credential is correct')
                print(e)
            except Exception as e:
                print(e)


    cap.release()
    out.release()
    cv2.destroyAllWindows()



def object_main():


    st.title("Flux traffic")
    st.write("Dans ce projet, nous allons détecter et suivre les voitures sur la route, et compter le nombre de véhicules circulant sur une route. Et les données seront stockées pour analyser différents véhicules qui circulent sur la route.")

    st.write()

    video_file_buffer = st.sidebar.file_uploader("Charger une vidéo", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    demo_video = 'data/Road.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    if video_file_buffer != None:
        vid = video_file_buffer.name
        with open(vid, mode='wb') as f:
            f.write(video_file_buffer.read())  # save video to disk

        st_video = open(vid, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        # video_file = 'street.mp4'
        cap = cv2.VideoCapture(vid)

        realTime(vid, cap)
        try:
            clip = moviepy.VideoFileClip('data/ouput/detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4', 'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video")

            # tffile.name = 'data/ouput/detected_video.mp4'
            # output_vid = open(tffile.name, 'rb')
            # read_output = output_vid.read()
            # st.text("vidéo de sotie")
            # st.video(read_output)

            # else:
            #     vid = cv2.VideoCapture(demo_video)
            #     tffile.name = demo_video
            #     demo_vid = open(tffile.name, 'rb')
            #     read_demo = demo_vid.read()
            #     st.text("vidéo d'entré")
            #     st.video(read_demo)
            #     realTime(demo_video, vid)
            #
            #     tffile.name ='data/ouput/detected_video.mp4'
            #     output_vid = open(tffile.name, 'rb')
            #     read_output = output_vid.read()
            #     st.text("vidéo de sotie")
            #     st.video(read_output)
        except OSError:
            'e'


if __name__ == '__main__':
    object_main()
