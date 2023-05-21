import requests
from PIL import Image
import json
import base64
import numpy as np
import cv2
import re
import os
import glob
from tqdm import tqdm
import math
import time
from gradio.processing_utils import encode_pil_to_base64

def decode_image(src):
    """
    解码图片
    :param src: 图片编码
    :return: str 保存到本地的文件名
    """
    # 1、信息提取
    result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", src, re.DOTALL)
    if result:
        ext = result.groupdict().get("ext")
        data = result.groupdict().get("data")

    else:
        raise Exception("Do not parse!")

    # 2、base64解码
    img = base64.urlsafe_b64decode(data)

    return img
    # 3、二进制文件保存
    # with open(save_path, "wb") as f:
    #     f.write(img)


def img2img(imgpath, url, prompt, width, height):
    imgse64 = encode_pil_to_base64(Image.open(imgpath))
    payload = {
        "init_images": [
            imgse64
        ],
        "resize_mode": 0,
        "denoising_strength": 0.75,
        "mask": None,
        "mask_blur": 4,
        "inpainting_fill": 1,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "inpainting_mask_invert": 0,
        "prompt": "masterpiece, best quality, " + prompt,
        "styles": [
            ""
        ],
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "batch_size": 8,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "restore_faces": False,
        "tiling": False,
        "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "eta": 0,
        "s_churn": 0,
        "s_tmax": 0,
        "s_tmin": 0,
        "s_noise": 1,
        "override_settings": {},
        "sampler_index": "Euler a",
        "include_init_images": False
        }

    req = requests.post(url = url, json = payload)
    return json.loads(req.text)

def text2img(url, prompt):
    payload = {
    "enable_hr": False,
    "denoising_strength": 0,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 2,
    "hr_upscaler": "string",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "prompt": prompt,
    "styles": [
        ""
    ],
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 50,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": False,
    "tiling": False,
    "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    "eta": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "script_args": [],
    "sampler_index": "Euler",
    }
    req = requests.post(url = url, json = payload)
    return json.loads(req.text)

def img2img_exapmle():
    src_img_path = "random90"
    prompt_path = src_img_path + "_prompt"
    gn_imgs_path = src_img_path + "_gn_4"
    if gn_imgs_path not in os.listdir():
        os.mkdir(gn_imgs_path)
    imgs_path = glob.glob(src_img_path + "/*")
    img2img_url = "http://127.0.0.1:8083/sdapi/v1/img2img"

    imgs_num = len(imgs_path)
    for img_idx, img_path in enumerate(imgs_path):
        print("SD生成进度: %.2f%%" % ((img_idx + 1) / float(imgs_num) * 100))
        gn_img_fname = img_path.split("/")[-1].split(".")[0]
        prompt_txt_path = prompt_path + "/" + img_path.split("/")[-1].split(".")[0] + ".txt"
        prompt = ""
        with open(prompt_txt_path, "r") as f:
            prompt = f.readline().strip()
        src_img = cv2.imread(img_path)
        h, w, c = src_img.shape
        height = 768
        width = 768
        if h > w:
            ratio = float(w) / float(h)
            width = math.ceil(width * ratio / 64.0) * 64
        elif h < w:
            ratio = float(h) / float(w)
            height = math.ceil(height * ratio / 64.0) * 64
    
        res = img2img(img_path, img2img_url, prompt, width, height)
        for idx, image in enumerate(res['images']):
            gn_img = decode_image(image)
            gn_img_save_path = gn_imgs_path + "/" + gn_img_fname + "_" + str(idx + 1) + ".jpg"
            with open(gn_img_save_path, "wb") as f:
                f.write(gn_img)

def text2img_example():
    # api 请求地址，server 启动命令 python launch.py --api 增加一个api参数
    txt2img_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    now = time.time()
    prompt = "1girl, beautiful, car, simple background"
    res = text2img(txt2img_url, prompt)
    end = time.time()
    print(end-now)
    gn_img_fname = "./"
    # print(res)
    for idx, image in enumerate(res['images']):
        # print(image)
        img = base64.urlsafe_b64decode(image)
        # gn_img = decode_image(image)
        # print(gn_img)
        gn_img_save_path = str(idx) + ".jpg"
        with open(gn_img_save_path, "wb") as f:
            f.write(img)

if __name__ == "__main__":
    text2img_example()
