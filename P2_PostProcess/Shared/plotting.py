from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from natsort import natsorted, ns
import settings

def image_list2pdf(image_list, labels, save_path, res=100):
    """
    :param image_list: list of images to concatenate into a pdf
    :param labels: singular string or array-like, if array-like, image must match length of image_list
    :param save_path: location to save pdf
    :param res: resolution of each pdf page
    :return:
    """
    if isinstance(labels, str):
        labels = np.repeat(labels, len(image_list)).tolist()

    im_list=[]
    first = True
    for im_path, label in zip(image_list, labels):
        if first:
            im1 = open_image(im_path, label=label)
            first = False
        else:
            im_list.append(open_image(im_path,label=label))

    if len(image_list)>0: # if any images exist
        pdf_filename = save_path + ".pdf"
        im1.save(pdf_filename, "PDF", resolution=res, save_all=True, append_images=im_list)


def open_image(im_path, label=None, margin=100, size=None):
    rgba = Image.open(im_path)
    if size is not None:
        width, height = size
    else:
        width, height = rgba.size

    # use a truetype font
    font = ImageFont.truetype(settings.PIL_fontstyle_path, 15)

    draw = ImageDraw.Draw(rgba)
    textwidth, textheight = draw.textsize(label)
    x = width - textwidth - margin
    y = height - textheight - margin
    if label is not None:
        fontsize = 1  # starting font size
        # portion of image width you want text width to be
        img_fraction = 0.3
        font = ImageFont.truetype(settings.PIL_fontstyle_path, fontsize)
        while font.getsize(label)[0] < img_fraction * rgba.size[0]:
            fontsize += 1
            font = ImageFont.truetype(settings.PIL_fontstyle_path, fontsize)
        draw.text((0, 0), label, (0, 0, 0), font=font)

    #font = ImageFont.truetype("sans-serif.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    #draw.text((0, 0),im_path.split("/")[-1],(0,0,0)) # this will draw text with Blackcolor and 16 size

    if size is not None:
        rgb = Image.new('RGB', size, (255, 255, 255))
    else:
        rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
    rgb.paste(rgba, mask=rgba.split()[3])               # paste using alpha channel as mask
    return rgb


def make_pdf_from_recording(recording_path, processed_folder_name):
    folder_ordered = []
    folder_ordered.append(recording_path + "/" + processed_folder_name + "/trace_view")
    folder_ordered.append(recording_path + "/" + processed_folder_name + "/lfp_view")
    for sorterName in settings.list_of_named_sorters:
        folder_ordered.append(recording_path+"/"+processed_folder_name + "/" + sorterName + "/spike_view")

    # crawl recording folder for any other folders with images
    for root, dir_names, file_names in os.walk(recording_path+"/"+processed_folder_name):
        for f in file_names:
            if (f.endswith(".png") and (root not in folder_ordered)):
                folder_ordered.append(root)

    image_list = []
    image_labels = []
    for full_folder_path in folder_ordered:
        folder_name_from_processed = full_folder_path.split(processed_folder_name)[-1]

        if os.path.isdir(full_folder_path):
            img_names = [f for f in listdir(full_folder_path) if (isfile(join(full_folder_path, f)) & (f.endswith(".png")))]
            img_names = natsorted(img_names, key=lambda y: y.lower())

            for img_name in img_names:
                image_list.append(full_folder_path+"/"+img_name)
                image_labels.append(folder_name_from_processed+"/"+img_name)

    image_list2pdf(image_list, image_labels, save_path=recording_path+"/"+processed_folder_name+"/summary_"+str(os.path.basename(recording_path)))
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


if __name__ == '__main__':
    main()