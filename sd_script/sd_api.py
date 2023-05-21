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
