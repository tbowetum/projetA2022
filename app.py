import tempfile

import cv2
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import streamlit as st
from vehicle_count import *

classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

def main():
    st.title('Flux de circulation')

    st.sidebar.title('settings')
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-test = "stSidebar][aria-expanded="true"] > div: first-child{width:400px;} 
    [data-test = "stSidebar][aria-expanded="false"] > div: first-child{width:400px; margin-left: -400px}
    </style>
    """,
    unsafe_allow_html=True,
    )

    ##case à cocher
    st.sidebar.markdown('---')
    save_vidéo = st.sidebar.checkbox('save video')
    Class_names = st.sidebar.checkbox('Classe')
    classe_id =[]

    classes = classNames

    ##liste des la
    if Class_names:
        class_list = st.sidebar.multiselect('selectionner la classe', list(classes), default='car')
        for each in class_list:
            classe_id.append(classes.index(each))

    ##chargement de la video
    video_file_buffer = st.sidebar.file_uploader("Charger une vidéo", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    demo_video = 'data/Highway.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix ='.mp4', delete=False)

    if not video_file_buffer:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
        demo_vid = open(tffile.name, 'rb')
        read_demo = demo_vid.read()

        st.sidebar.text("vidéo d'entré")
        st.sidebar.video(read_demo)
    else:
        tffile.write(video_file_buffer.read())
        demo_vid = open(tffile.name, 'rb')
        read_demo = demo_vid.read()

        st.sidebar.text("vidéo d'entré")
        st.sidebar.video(read_demo)

    print(tffile)
    st.sidebar.markdown('---')

    kpi, kpi1 = st.beta_columns(2)

    with kpi:
        st.markdown("**Number of cars**")
        kpi_text = st.markdown("0")

    with kpi1:
        st.markdown("**traffic dessity**")
        kpi_text = st.markdown("0")




if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

