from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd

import settings

def image_list2pdf(image_list, save_path, label=None, res=300):
    im_list=[]
    first = True
    for im_path in image_list:
        if first:
            im1 = open_image(im_path, label=label)
            first = False
        else:
            im_list.append(open_image(im_path,label=label))
    pdf_filename = save_path+".pdf"
    im1.save(pdf_filename, "PDF", resolution=res, save_all=True, append_images=im_list)


def open_image(im_path, label, margin=10):
    rgba = Image.open(im_path)
    width, height = rgba.size
    draw = ImageDraw.Draw(rgba)
    textwidth, textheight = draw.textsize(im_path.split("/")[-1])
    x = width - textwidth - margin
    y = height - textheight - margin
    if label is not None:
        draw.text((x, y), label)
    draw.text((0, 0),im_path.split("/")[-1],(0,0,0)) # this will draw text with Blackcolor and 16 size

    rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
    rgb.paste(rgba, mask=rgba.split()[3])               # paste using alpha channel as mask
    return rgb


def make_pdf_from_recording(recording_path, processed_folder_name):
    subdirs = [f.path for f in os.scandir(recording_path+"/"+processed_folder_name) if f.is_dir()]

    folder_order = ["trace_view", "lfp_view"]
    if mountainsort

    # get a list of images to include in a pdf
    for dir in [f.path for f in os.scandir(recording_path+"/"+processed_folder_name) if f.is_dir()]:


        if dir.split("/")[-1] in settings.list_of_named_sorters:


    image_list2pdf(image_list, save_path)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


if __name__ == '__main__':
    main()