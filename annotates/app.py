import os
import os.path as osp
import streamlit as st
import json
import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

import ast
import requests
from io import BytesIO
from icecream import ic


### Image utility functions
def load_image(
    path_to_image: str,
    backend: str = "cv2",
    toRGB: bool = True,
    to_array: bool = True,
):
    """Loading image from specied path

    Args:
        path_to_image (str): absolute paths to images
        toRGB (bool, optional): _description_. Defaults to True.

    Returns:
        (Any): output image
    """
    image = None

    if backend == "cv2":
        image = cv2.imread(path_to_image)
        if toRGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif backend == "pil":
        image = Image.open(path_to_image)
        if to_array:
            image = np.array(image)

    return image


def expand2square_cv2(
        img,
        fill=255
):
    """
    From https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/

    Add padding to the short side to 
    make the image square while maintaining 
    the aspect ratio of the rectangular image.
    """
    height, width, _ = img.shape
    if width == height:
        return img
    elif width > height:
        result = np.full((width, width, 3), fill)
        pad_margin = int((width - height) // 2)
        result[pad_margin:-pad_margin, :, :] = img
        return result
    else:
        result = np.full((height, height, 3), fill)
        pad_margin = int((height - width) // 2)
        result[:, pad_margin:-pad_margin, :] = img
        return result


### UI functions
def add_variables(**kwargs):
    st.session_state.idx = 0
    st.session_state.current_fns = None
    st.session_state.kw_pattern = re.compile(r'(\d+:\d+\s*)+')
    st.session_state.list_pattern = re.compile(r'(\d+\s*)+')

    for k, v in kwargs.items():
        if k not in st.session_state:
            st.session_state[k] = v

    annt_file = st.session_state.annt_file
    if annt_file and osp.exists(annt_file):
        st.session_state.annt_dict = json.load(open(annt_file, 'r'))
    else:
        st.session_state.annt_dict = {}

    st.session_state.img_fns = os.listdir(st.session_state.image_dir)


def process_intput(text):
    text = text.lower().strip()
    fns = st.session_state.current_fns
    kw_texts = st.session_state.kw_pattern.match(text)

    if text in ['s', 'e']:
        with open(st.session_state.annt_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.annt_dict, f, indent=4)
        f.close()

    # Back to previous batch of images
    elif text == 'p':
        st.session_state.idx -= st.session_state.batch_size        

    elif kw_texts:
        kw_texts = kw_texts.group(0).strip().split(' ')
        for kw in kw_texts:
            idx, annt = kw.split(':')
            st.session_state.annt_dict[fns[int(idx)-1]] = st.session_state.labels[int(annt)-1]

    elif st.session_state.list_pattern.match(text):
        annts = text.split(' ')

        for fn, annt in zip(fns, annts):
            st.session_state.annt_dict[fn] = st.session_state.labels[int(annt)-1]


def display_multiple_images(imgs, img_size=224, nrows=2):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # org 
    org = (4, 24)  # w, h

    # fontScale 
    fontScale = 1

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 2

    num_imgs_each_row = int(len(imgs) / nrows)
    new_sizes = (img_size, img_size)

    idx = 0
    row_imgs = []

    while idx < len(imgs):
        row_img = []
        batch_imgs = imgs[idx:idx+num_imgs_each_row]
        for i, img in enumerate(batch_imgs):
            img_plot_idx = idx + i + 1
            img = expand2square_cv2(img)
            if img_size is not None:
                img = cv2.resize(img, new_sizes)

            img = cv2.putText(img, str(img_plot_idx) , org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)                     
            
            row_img.append(img)

        row_img = np.hstack(row_img)
        row_imgs.append(row_img)
        idx += num_imgs_each_row

    grid = np.vstack(row_imgs)
    st.image(grid)


def display_annotation_ui():
    col1, _, col3 = st.columns([0.5, 0.1, 0.4], gap="large")

    st.write("Number of images:", len(st.session_state.img_fns))

    labels_display = \
    """
    \t (key → option)
    """
    for i, label in enumerate(st.session_state.labels):
        labels_display += \
         f"""
         \t{i+1} → {label}
         """

    col1.write(
        ":green[***Command Map***]:\n" +
        labels_display + 
        """
        \tP → [Previous]\n
        \tS → [Save the process]\n
        \tE → [Save & Exit]\n
        """        
    )

    st.session_state.current_fns = st.session_state.img_fns[st.session_state.idx : st.session_state.idx + st.session_state.batch_size]
    fps = [osp.join(st.session_state.image_dir, fn) for fn in st.session_state.current_fns]
    imgs = [load_image(fp) for fp in fps]

    with col3:
        st.write(":green[***Image***]")
        display_multiple_images(imgs)


def submit():
    user_input = st.session_state.text
    if user_input:
        process_intput(user_input)

        if user_input[0].isdigit():
            # Move to next data item
            st.session_state.idx += st.session_state.batch_size

        # Clear screen
        st.empty()
        st.session_state.text = ''

    display_annotation_ui()


def text_input():
    current_label = ""
    for i, fn in enumerate(st.session_state.current_fns):
        label = st.session_state.annt_dict[fn]
        current_label += label
        if i != st.session_state.batch_size-1:
            current_label += ' '

    st.text_input(
        f"Current label is :green[{current_label}]. Enter your command:",
        key="text",
        on_change=submit
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image-dir")
    parser.add_argument("-l", "--labels", nargs='+', default=["red", "green", "blue"])
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("-f", "--annt-file", type=str, default="test.json")
    parser.add_argument("-r", "--rerun", action="store_true")        
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    st.set_page_config(
        page_title="Streamlit Classification Annotation App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.extremelycoolapp.com/help",
            "Report a bug": "https://www.extremelycoolapp.com/bug",
            "About": "# This is a header. This is an *extremely* cool app!",
        },
    )

    if "idx" not in st.session_state:
        add_variables(**vars(args))
        display_annotation_ui()
    elif args.rerun:
        add_variables(**vars(args))
        display_annotation_ui()            

    text_input()
