"""
Refer:
https://github.com/RVC-Boss/GPT-SoVITS
"""

import os
import traceback,gradio as gr
import logging
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

logger = logging.getLogger(__name__)
import librosa
import soundfile as sf
import torch
import sys
 
parent_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(parent_directory, "GPT_SoVITS"))
sys.path.append(os.path.join(parent_directory, "tools"))
sys.path.append(os.path.join(parent_directory, "tools/uvr5"))
 
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
 
from tools.uvr5.mdxnet import MDXNetDereverb
from tools.uvr5.vr import AudioPre, AudioPreDeEcho
import webbrowser
 
# 添加新的环境变量
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'
 
weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))
 
uvr5_choose_map = {
    i18n("音频不带和声，只有人声和背景音"): "HP3_all_vocals",
    i18n("音频带和声，只提取主人声"): "HP5_only_main_vocal",
}
 
def contains_english_letters(input_str):
    # 使用isalpha()方法检查字符串是否只包含字母，并且使用isascii()方法确保只包含ASCII字符
    return any(char.isalpha() for char in input_str if char.isascii())
 
from config import infer_device, is_half
 
def uvr(model_choose, device, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    model_name = uvr5_choose_map[i18n(model_choose)]
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg = int(agg),
                model_path = os.path.join(
                    weight_uvr5_root, model_name + ".pth"
                ),
                device = device,
                is_half = is_half,
            )
        is_hp3 = "HP3" in model_name
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            # inp_path = os.path.join(inp_root, path)
            inp_path = path
            if(os.path.isfile(inp_path)==False):continue
            try:
                done = 0
                try:
                    y, sr = librosa.load(inp_path, sr=None)
                    info = sf.info(inp_path)
                    channels = info.channels
                    if channels == 2 and sr == 44100:
                        need_reformat = 0
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                        )
                        done = 1
                    else:
                        need_reformat = 1
                except:
                    need_reformat = 1
                    traceback.print_exc()
                if need_reformat == 1:
                    tmp_path = "%s/%s.reformatted.wav" % (
                        os.path.join(os.environ["TEMP"]),
                        os.path.basename(inp_path),
                    )
                    y_resampled = librosa.resample(y, sr, 44100)
                    sf.write(tmp_path, y_resampled, 44100, "PCM_16")
                    inp_path = tmp_path
                try:
                    if done == 0:
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    # yield "\n".join(infos)
                except:
                    try:
                        if done == 0:
                            pre_fun._path_audio_(
                                inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                            )
                        infos.append("%s->Success" % (os.path.basename(inp_path)))
                        # yield "\n".join(infos)
                    except:
                        infos.append(
                            "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                        )
                        # yield "\n".join(infos)
            except:
                infos.append("Oh my god. %s->%s"%(os.path.basename(inp_path), traceback.format_exc()))
                # yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        # yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
    # yield "\n".join(infos)
    return infos
 
# slice_inp_path
 
import numpy as np
from scipy.io import wavfile
from my_utils import load_audio
from tools.slicer2 import Slicer
 
def slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha):
    os.makedirs(opt_root,exist_ok=True)
    if os.path.isfile(inp):
        inputs=[inp]
    elif os.path.isdir(inp):
        inputs=[os.path.join(inp, name) for name in sorted(list(os.listdir(inp)))]
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    slicer = Slicer(
        sr=32000,  # 长音频采样率
        threshold=      int(threshold),  # 音量小于这个值视作静音的备选切割点
        min_length=     int(min_length),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval=   int(min_interval),  # 最短切割间隔
        hop_size=       int(hop_size),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept=   int(max_sil_kept),  # 切完后静音最多留多长
    )
    _max=float(_max)
    alpha=float(alpha)
    for inp_path in inputs:
        print(inp_path)
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            print(audio.shape)
            for chunk, start, end in slicer.slice(audio):  # start和end是帧数
                tmp_max = np.abs(chunk).max()
                if(tmp_max>1):chunk/=tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%s_%s.wav" % (opt_root, name, start, end),
                    32000,
                    # chunk.astype(np.float32),
                    (chunk * 32767).astype(np.int16),
                )
        except:
            print(inp_path,"->fail->",traceback.format_exc())
    return "执行完毕，请检查输出文件"
 
def asr(asr_inp_dir, asr_opt_dir):
    print(asr_inp_dir)
    print(asr_opt_dir)
    from funasr import AutoModel
    from modelscope.utils.constant import Tasks
    
    opt_name = os.path.basename(asr_inp_dir)
 
    path_asr='tools/damo_asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
    path_vad='tools/damo_asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
    path_punc='tools/damo_asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
    path_asr=path_asr if os.path.exists(path_asr)else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    path_vad=path_vad if os.path.exists(path_vad)else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    path_punc=path_punc if os.path.exists(path_punc)else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
 
    model = AutoModel(model=path_asr, model_revision="v2.0.4",
                      vad_model=path_vad,
                      vad_model_revision="v2.0.4",
                      punc_model=path_punc,
                      punc_model_revision="v2.0.4",
                      )
 
    print(asr_inp_dir)
    opt = []
    for name in os.listdir(asr_inp_dir):
        try:
            tmp = model.generate(input="%s/%s"%(asr_inp_dir, name))
            print(tmp)
            text = tmp[0]["text"]
            opt.append("%s/%s|%s|ZH|%s"%(asr_inp_dir, name, opt_name, text))
        except:
            print(traceback.format_exc())
 
    os.makedirs(asr_opt_dir, exist_ok = True)
    with open("%s/%s.list"%(asr_opt_dir, opt_name), "w", encoding="utf-8") as f:f.write("\n".join(opt))
 
ps1abc = []
import my_utils
from subprocess import Popen
import subprocess
import platform
import psutil
import signal
 
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
from time import time as ttime
import shutil
 
def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir_ = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s/%s.pth" % (dir_, ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir_, name))
 
def get_text_1(inp_text, opt_dir, bert_pretrained_dir, is_half, txt_path):
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    device = "cuda:0"
    print(bert_pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
 
    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
 
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
 
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
 
        return phone_level_feature.T
 
    def process(data, res):
        for name, text, lan in data:
            try:
                name = os.path.basename(name)
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan
                )
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if os.path.exists(path_bert) == False and lan == "zh":
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)
                phones = " ".join(phones)
                # res.append([name,phones])
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())
 
    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
 
    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
    }
    for line in lines:
        try:
            wav_name, spk_name, language, text = line.split("|")
            # todo.append([name,text,"zh"])
            todo.append(
                [wav_name, text, language_v1_to_language_v2.get(language, language)]
            )
        except:
            print(line, traceback.format_exc())
 
    process(todo, res)
    
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
 
        
from feature_extractor import cnhubert
 
def get_hubert_wav32k(inp_text, inp_wav_dir, opt_dir, ssl_pretrained_dir, is_half):
    hubert_dir = "%s/4-cnhubert" % (opt_dir)
    wav32dir = "%s/5-wav32k" % (opt_dir)
    os.makedirs(opt_dir, exist_ok = True)
    os.makedirs(hubert_dir, exist_ok = True)
    os.makedirs(wav32dir, exist_ok = True)
    maxx = 0.95
    alpha = 0.5
    device = "cuda:0"
    cnhubert.cnhubert_base_path = ssl_pretrained_dir
    model = cnhubert.get_model()
    if(is_half == True):
        model = model.half().to(device)
    else:
        model = model.to(device)
        
    nan_fails=[]
    def name2go(wav_name):
        hubert_path="%s/%s.pt"%(hubert_dir, wav_name)
        if(os.path.exists(hubert_path)):return
        wav_path="%s/%s"%(inp_wav_dir,wav_name)
        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.2:
            print("%s-filtered" % (wav_name, tmp_max))
            return
        tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
        tmp_audio = librosa.resample(
            tmp_audio32b, orig_sr=32000, target_sr=16000
        )#不是重采样问题
        tensor_wav16 = torch.from_numpy(tmp_audio)
        if (is_half == True):
            tensor_wav16=tensor_wav16.half().to(device)
        else:
            tensor_wav16 = tensor_wav16.to(device)
        ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
        if np.isnan(ssl.detach().numpy()).sum()!= 0:
            nan_fails.append(wav_name)
            print("nan filtered:%s"%wav_name)
            return
        wavfile.write(
            "%s/%s"%(wav32dir,wav_name),
            32000,
            tmp_audio32.astype("int16"),
        )
        my_save(ssl,hubert_path )
        
    with open(inp_text,"r",encoding="utf8")as f:
        lines=f.read().strip("\n").split("\n")
 
    for line in lines:
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name=os.path.basename(wav_name)
            name2go(wav_name)
        except:
            print(line,traceback.format_exc())
 
    if(len(nan_fails)>0 and is_half==True):
        is_half=False
        model=model.float()
        for wav_name in nan_fails:
            try:
                name2go(wav_name)
            except:
                print(wav_name,traceback.format_exc())
 
from module.models import SynthesizerTrn      
import utils
def get_semantic(inp_text, opt_dir, s2config_path, pretrained_s2G, is_half):
    hubert_dir = "%s/4-cnhubert" % (opt_dir)
    semantic_path = "%s/6-name2semantic.tsv" % (opt_dir)
    if os.path.exists(semantic_path) == False:
        os.makedirs(opt_dir, exist_ok=True)
    device = "cuda:0"
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    
    vq_model.load_state_dict(
        torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False
    )
    # print(
    #     vq_model.load_state_dict(
    #         torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False
    #     )
    # )
    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))
 
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
 
    lines1 = []
    for line in lines:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
 
        
import json
import yaml
 
pretrained_sovits_name="/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append(name)
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append(name)
    return SoVITS_names,GPT_names
SoVITS_weight_root = "/openbayes/home/GPT-SoVITS/SoVITS_weights"
GPT_weight_root = "/openbayes/home/GPT-SoVITS/GPT_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
os.makedirs(GPT_weight_root,exist_ok=True)
SoVITS_names, GPT_names = get_weights_names()
 
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
 
p_train_SoVITS = None      
 
def kill_proc_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return
 
    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
 
system = platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)
 
def open1Ba(batch_size,total_epoch,exp_root,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D):
    global p_train_SoVITS
    if(p_train_SoVITS==None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,batch_size//2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root
        data["name"]=exp_name
        tmp_config_path="%s/tmp_s2.json"%tmp
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))
        
        python_exec = sys.executable
        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%(python_exec,tmp_config_path)
        #yield "SoVITS训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print("SoVITS训练开始")
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS=None
        #yield "SoVITS训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        print("SoVITS训练完成")
    else:
        print("已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务")
        #yield "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}
 
p_train_GPT = None
def open1Bb(batch_size,total_epoch,exp_root,exp_name,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1):
    global p_train_GPT
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["half_weights_save_dir"]=GPT_weight_root
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir
 
        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_numbers.replace("-",",")
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%tmp
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        python_exec = sys.executable
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        # yield "GPT训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print("GPT训练开始")
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        # yield "GPT训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        print("GPT训练完成")
    else:
        #yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print("已有正在进行的GPT训练任务，需先终止才能开启下一次任务")
 
import glob
import re
p_tts_inference=None        
def change_tts_inference(if_tts,charactor_name,bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path,is_half,webui_port_infer_tts,is_share):
    global p_tts_inference
    if(if_tts==True and p_tts_inference==None):
        os.environ["gpt_path"]=gpt_path if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
        os.environ["sovits_path"]=sovits_path if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
        os.environ["charactor_name"]=charactor_name
        os.environ["cnhubert_base_path"]=cnhubert_base_path
        os.environ["bert_path"]=bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_number
        os.environ["is_half"]=str(is_half)
        os.environ["infer_ttswebui"]=str(webui_port_infer_tts)
        os.environ["is_share"]=str(is_share)
        
        python_exec = sys.executable
        print(i18n("TTS推理进程已开启"))
        cmd = [python_exec, "GPT_SoVITS/inference_webui.py"]
        # 使用 Popen 启动一个子进程，运行一个命令，比如 ls 命令
        p_tts_inference = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 获取输出
        output, _ = p_tts_inference.communicate()
        print(output)
        #p_tts_inference = Popen(cmd, shell=True)
    elif(if_tts==False and p_tts_inference!=None):
        kill_process(p_tts_inference.pid)
        p_tts_inference=None
        print(i18n("TTS推理进程已关闭"))
 
def train_btn(charactor_name, model_choose, raw_audio_input_root):
 
    # 1、提取人声，去背景音
    # charactor_name = "nup"
    # raw_audio_input_root = "/openbayes/input/input0/"
    
    uvr5_output = "output/uvr5_opt/" + charactor_name
    if not os.path.exists(uvr5_output):
        os.makedirs(uvr5_output)
 
    uvr5_output_vocal = uvr5_output + "/" + "vocal"
    if not os.path.exists(uvr5_output_vocal):
        os.makedirs(uvr5_output_vocal)
    paths = ""
 
    uvr5_output_ins = uvr5_output + "/" + "ins"
    if not os.path.exists(uvr5_output_ins):
        os.makedirs(uvr5_output_ins)
    uvr5_agg = 10
    uvr5_format0 = "wav"
    uvr_infos = []
    uvr_infos = uvr(model_choose, infer_device, raw_audio_input_root, uvr5_output_vocal, paths, uvr5_output_ins, uvr5_agg, uvr5_format0)
    # yield uvr_infos
    uvr5_vocals = glob.glob(os.path.join(uvr5_output_vocal, "*.wav"))
    for uvr5_vocal in uvr5_vocals:
        if "instrument" in uvr5_vocal:
            fname = uvr5_vocal.split("/")[-1]
            shutil.move(uvr5_vocal, os.path.join(uvr5_output_ins, fname))
            
    uvr5_inss = glob.glob(os.path.join(uvr5_output_ins, "*.wav"))
    for uvr5_ins in uvr5_inss:
        if "vocal" in uvr5_ins:
            fname = uvr5_ins.split("/")[-1]
            shutil.move(uvr5_ins, os.path.join(uvr5_output_vocal, fname))
            
    yield "第一步，音频处理完成"
 
    # 2、音频切分为小段
    slice_inp = uvr5_output_vocal
    slice_opt = "output/slicer_opt/" + charactor_name
    if not os.path.exists(slice_opt):
        os.makedirs(slice_opt)
    slice_threshold = -34
    slice_min_length = 4000
    slice_min_interval = 300
    slice_hop_size = 10
    slice_max_sil_kept = 500
    slice_max = 0.9
    slice_alpha = 0.25
    slice(slice_inp, slice_opt, slice_threshold, slice_min_length, slice_min_interval, slice_hop_size, slice_max_sil_kept, slice_max, slice_alpha)
    yield "第二步，音频切分完成"
 
    # 3、中文音频识别
    asr_inp_dir = slice_opt
    asr_opt_dir = "output/asr_opt/" + charactor_name
    if not os.path.exists(asr_opt_dir):
        os.makedirs(asr_opt_dir)
    asr(asr_inp_dir, asr_opt_dir)
    yield "第三步，音频识别完成"
 
    # 4、去掉英文
    process_list_path = os.path.join(asr_opt_dir, charactor_name + ".list")
    with open(process_list_path, "r", encoding = "utf-8") as f:
        speaker_list = f.readlines()
        speaker_datas = list(map(lambda x:x.strip().split("|")[-1], speaker_list))
 
    with open(process_list_path, "w", encoding = "utf-8") as f:
        for idx, data in enumerate(speaker_datas):
            if contains_english_letters(data):
                pass
            else:
                f.write(speaker_list[idx].split("/")[-1])
    yield "第四步，去除英文完成"
    
    # 5、数据 token 化
    token_inp_text = os.path.join(asr_opt_dir, charactor_name + ".list")
    token_inp_wav_dir = slice_opt
    token_exp_name = charactor_name
    token_gpu_numbers1a = "0-0"
    token_gpu_numbers1Ba = "0-0"
    token_gpu_numbers1c = "0-0"
    token_bert_pretrained_dir = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    token_cnhubert_base_dir = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"
    token_pretrained_s2G = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/s2G488k.pth"
    token_s2config_path = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/configs/s2.json"
    
    token_exp_root = "logs"
    token_bert_is_half = False
    token_opt_dir = "%s/%s"%(token_exp_root, token_exp_name)
    os.makedirs(token_opt_dir, exist_ok = True)
    token_path_text = "%s/2-name2text.txt" % token_opt_dir
    if(os.path.exists(token_path_text)==False or (os.path.exists(token_path_text)==True and len(open(token_path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
        get_text_1(token_inp_text, token_opt_dir, token_bert_pretrained_dir, token_bert_is_half, token_path_text)
    get_hubert_wav32k(token_inp_text, token_inp_wav_dir, token_opt_dir, token_cnhubert_base_dir, token_bert_is_half)
    get_semantic(token_inp_text, token_opt_dir, token_s2config_path, token_pretrained_s2G, token_bert_is_half)
    yield "第五步，数据 token 化完成"
    
    # 6、模型微调训练
    train_exp_root = token_exp_root
    train_batch_size = 12
    train_total_epoch = 15
    train_exp_name = charactor_name
    train_text_low_lr_rate = 0.4
    train_if_save_latest = True
    train_if_save_every_weights = True
    train_save_every_epoch = 4
    train_gpu_numbers = "0"
    train_pretrained_s2G = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/s2G488k.pth"
    train_pretrained_s2D = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/s2D488k.pth"
    train_pretrained_s1 = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    
    open1Ba(train_batch_size, train_total_epoch, train_exp_root, train_exp_name, train_text_low_lr_rate, train_if_save_latest, \
            train_if_save_every_weights, train_if_save_every_weights, train_gpu_numbers, train_pretrained_s2G, train_pretrained_s2D)
    open1Bb(train_batch_size, train_total_epoch, train_exp_root, train_exp_name, train_if_save_latest, train_if_save_every_weights, train_save_every_epoch, train_gpu_numbers, train_pretrained_s1)
    yield "第六步，模型微调训练完成"
    
    # 7、模型预测
    infer_if_tts = True
    infer_bert_pretrained_dir = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    infer_cnhubert_base_dir = "/openbayes/home/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"
    infer_gpu_number_1C = "0"
    infer_is_half = False
    infer_webui_port_infer_tts = 8080
    infer_is_share = True
    infer_sovits_paths = glob.glob("/openbayes/home/GPT-SoVITS/SoVITS_weights/{}_*.pth".format(charactor_name))
    max_num = -1
    max_file = ''
    for file in infer_sovits_paths:
        # 使用正则表达式从文件名中提取'e'后面的数字
        match = re.search(r'%s_e(\d+)_' % charactor_name, file)
        if match:
            num = int(match.group(1))  # group(1)表示第一个括号匹配的内容，即'e'后面的数字
            # 如果找到更大的数字，则更新最大数字和文件名
            if num > max_num:
                max_num = num
                max_file = file
    infer_sovits_path = max_file
    
    infer_GPT_dropdowns = glob.glob("/openbayes/home/GPT-SoVITS/GPT_weights/{}-*.ckpt".format(charactor_name))
    max_num = -1
    max_file = ''
    for file in infer_GPT_dropdowns:
        # 使用正则表达式从文件名中提取'e'后面的数字
        match = re.search(r'%s-e(\d+).' % charactor_name, file)
        if match:
            num = int(match.group(1))  # group(1)表示第一个括号匹配的内容，即'e'后面的数字
            # 如果找到更大的数字，则更新最大数字和文件名
            if num > max_num:
                max_num = num
                max_file = file
    infer_GPT_dropdown = max_file
    
    yield "模型正在开启预测，请稍后（1min左右）"
    change_tts_inference(infer_if_tts, charactor_name, infer_bert_pretrained_dir, infer_cnhubert_base_dir, infer_gpu_number_1C, infer_GPT_dropdown, infer_sovits_path, \
                        infer_is_half, infer_webui_port_infer_tts, infer_is_share)
    
    
if __name__ == "__main__":
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                charactor_name = gr.Textbox(label=i18n("*模型名"), value="test", interactive=True)
                raw_audio_input_root = gr.Textbox(label=i18n("*数据集地址"), value="/openbayes/input/input0", interactive=True)
                model_choose = gr.Dropdown(label = i18n("音频数据类型"), choices = list(uvr5_choose_map.keys()))
            with gr.Column():
                btn = gr.Button(i18n("开始训练"))
                text_output = gr.TextArea(
                    label = i18n("输出结果"),
                    lines = 2,
                    )
                btn.click(
                    train_btn, 
                    [
                        charactor_name,
                        model_choose,
                        raw_audio_input_root,
                    ],
                    [
                        text_output,
                    ]
                    ) 
    # webbrowser.open("http://127.0.0.1:8088")
    app.queue(concurrency_count = 5, max_size = 20).launch(server_name = "127.0.0.1", server_port = 8088, share = True)
