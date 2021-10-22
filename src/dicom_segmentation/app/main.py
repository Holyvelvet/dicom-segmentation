import base64
import os
import time

import pydicom as pdm
from pydicom.filebase import DicomBytesIO

import numpy as np
import cv2
import torch
from segmentation_models_pytorch.unet import Unet
from albumentations import Normalize

from PIL import Image

import streamlit as st

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Config
DEVICE = 'cpu'
model_weights_path = r'C:\Users\ALM\PycharmProjects\dicom-segmentation\data\06_models\best_model.pth'
model_input_path = r'C:\Users\ALM\PycharmProjects\dicom-segmentation\data\05_model_input'
model_output_path = r'C:\Users\ALM\PycharmProjects\dicom-segmentation\data\07_model_output'

threshold = 0.3


# Loading the model
model = Unet('efficientnet-b2', encoder_weights="imagenet", classes=3, activation=None)
model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
model.eval()


# Helper functions
def get_overlaid_masks_on_image(
        one_slice_image: np.ndarray,
        one_slice_mask: np.ndarray,
        w: float = 512,
        h: float = 512,
        dpi: float = 100,
        path_to_save: str = '/content/',
        name_to_save: str = 'img_name'):
    """overlap masks on image and save this as a new image."""

    path_to_save_ = os.path.join(path_to_save, name_to_save)
    lung, heart, trachea = [one_slice_mask[:, :, i] for i in range(3)]
    figsize = (w / dpi), (h / dpi)
    fig = plt.figure(figsize=figsize)
    fig.add_axes([0, 0, 1, 1])

    # image
    plt.imshow(one_slice_image, cmap="bone")

    # overlaying segmentation masks
    plt.imshow(np.ma.masked_where(lung == False, lung), cmap='cool', alpha=0.3)
    plt.imshow(np.ma.masked_where(heart == False, heart), cmap='autumn', alpha=0.3)
    plt.imshow(np.ma.masked_where(trachea == False, trachea), cmap='autumn_r', alpha=0.3)

    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)

    fig.savefig(f"{path_to_save_}.png", bbox_inches='tight', pad_inches=0, dpi=dpi, format="png")

    plt.close()

    return None


def convert_to_jpg(dcm, output_path, img_name, eps=1e-9):

    im = dcm.pixel_array
    im = (im.astype(np.float32) - im.min()) * 255.0 / (im.max() - im.min()) + eps
    im = im.astype(np.uint8)

    save_path = os.path.join(output_path, img_name + ".jpg")
    if not cv2.imwrite(save_path, im):
        raise Exception("Could not write image")

    cv2.destroyAllWindows()
    return None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Main process of the app
def run_process(*uploaded_files):
    # Create folders for data
    dirname = str(time.time_ns())
    input_data_path = os.path.join(model_input_path, dirname)
    output_data_path = os.path.join(model_output_path, dirname)

    os.makedirs(input_data_path, exist_ok=True)
    os.makedirs(output_data_path, exist_ok=True)

    # Sort the list in accordance to file names
    uploaded_files = sorted(uploaded_files, key=lambda x: int(x.name[:-4]))

    # Processing uploaded files
    st.info("Processing uploaded files.")
    progress_bar = st.progress(0)

    # Main loop of the program
    dcm_files = []
    images_file_path = []
    for file in uploaded_files:
        # Reading files
        raw = DicomBytesIO(file.getvalue())
        dcm = pdm.dcmread(raw)

        dcm_files.append(dcm)

        # Converting to jpeg - needed to feed the model
        file_name = file.name[:-4]
        convert_to_jpg(dcm, input_data_path, file_name)

        # Making predictions
        img_ = cv2.imread(os.path.join(input_data_path, file_name + '.jpg'))
        img = Normalize().apply(img_)
        tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
        prediction = model.forward(tensor.to(DEVICE))
        prediction = prediction.cpu().detach().numpy()
        prediction = prediction.squeeze(0).transpose(1, 2, 0)
        prediction = sigmoid(prediction)
        prediction = (prediction >= threshold).astype(np.float32)

        prediction = (prediction * 255).astype("uint8")

        get_overlaid_masks_on_image(img_,
                                    prediction,
                                    path_to_save=output_data_path,
                                    name_to_save=file_name)

        images_file_path.append(os.path.join(output_data_path, file_name + '.png'))

        progress_bar.progress(len(dcm_files) / len(uploaded_files))

    # Saving results into a gif
    output_file_path = os.path.join(output_data_path, 'output.gif')

    img, *imgs = [Image.open(f) for f in images_file_path]
    img.save(fp=output_file_path, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)

    # Show results as a gif
    file_ = open(output_file_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="Segmentation result">',
        unsafe_allow_html=True,
    )


# App
st.title('DICOM heart, lungs and trachea segmentation')

uploaded_files = st.file_uploader(
    "Choose a DICOM file",
    type='dcm',
    accept_multiple_files=True
)

if uploaded_files:
    st.button('Run the model on uploaded files', on_click=run_process, args=uploaded_files)
