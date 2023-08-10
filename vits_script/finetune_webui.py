# -*- coding:utf-8 -*-
"""
Author: Jack Cui
https://space.bilibili.com/331507846
"""
import gradio as gr
import webbrowser
import time
import os
import glob
import shutil
import subprocess as sp
import time
import json
import torch
import torchaudio
import whisper

import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import librosa
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

torch.backends.cudnn.benchmark = True
global_step = 0

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(train_loader)):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)
        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                epoch,
                100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
                }
                utils.summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:

                evaluate(hps, net_g, eval_loader, writer_eval)
                
                utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                    os.path.join(hps.model_dir, "G_latest.pth"))
                
                utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                    os.path.join(hps.model_dir, "D_latest.pth"))

            if hps.preserved > 0:
                utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                        os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                old_g = utils.oldest_checkpoint_path(hps.model_dir, "G_[0-9]*.pth",
                                                    preserved=hps.preserved)  # Preserve 4 (default) historical checkpoints.
                old_d = utils.oldest_checkpoint_path(hps.model_dir, "D_[0-9]*.pth", preserved=hps.preserved)
                if os.path.exists(old_g):
                    print(f"remove {old_g}")
                    os.remove(old_g)
                if os.path.exists(old_d):
                    print(f"remove {old_d}")
                    os.remove(old_d)

        global_step += 1
        if epoch > hps.max_epochs:
            print("Maximum epoch reached, closing training...")
            exit()
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))

def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step,
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


def get_hparams(continue_train, max_epochs, init=True):

    model_dir = "OUTPUT_MODEL"
    # model_dir = os.path.join("./", model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if continue_train == "是":
        drop_speaker_embed = False
        cont = True
    else:
        drop_speaker_embed = True
        cont = False

    config_path = os.path.join("configs", "modified_finetune_speaker.json")
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = utils.HParams(**config)
    hparams.model_dir = model_dir
    hparams.max_epochs = max_epochs
    hparams.cont = cont
    hparams.drop_speaker_embed = drop_speaker_embed
    hparams.train_with_pretrained_model = True
    hparams.preserved = 4
    return hparams

def train_main(continue_train, max_epochs):
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'

    hps = get_hparams(continue_train, max_epochs)
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

def run(rank, n_gpus, hps):
    global global_step
    symbols = hps['symbols']
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir = hps.model_dir)
        writer_eval = SummaryWriter(log_dir = os.path.join(hps.model_dir, "eval"))

    # Use gloo backend on Windows for Pytorch
    dist.init_process_group(backend = 'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, symbols)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,300,400,500,600,700,800,900,1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)
    
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, symbols)
        eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
            batch_size=hps.train.batch_size, pin_memory=True,
            drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    # load existing model
    if hps.cont:
        try:
            _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_latest.pth"), net_g, None)
            _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_latest.pth"), net_d, None)
            global_step = (epoch_str - 1) * len(train_loader)
        except:
            print("Failed to find latest checkpoint, loading G_0.pth...")
            if hps.train_with_pretrained_model:
                print("Train with pretrained model...")
                _, _, _, epoch_str = utils.load_checkpoint(os.path.join("pretrained_models", "G_0.pth"), net_g, None)
                _, _, _, epoch_str = utils.load_checkpoint(os.path.join("pretrained_models", "D_0.pth"), net_d, None)
            else:
                print("Train without pretrained model...")
            epoch_str = 1
            global_step = 0
    else:
        if hps.train_with_pretrained_model:
            print("Train with pretrained model...")
            _, _, _, epoch_str = utils.load_checkpoint(os.path.join("pretrained_models", "G_0.pth"), net_g, None)
            _, _, _, epoch_str = utils.load_checkpoint(os.path.join("pretrained_models", "D_0.pth"), net_d, None)
        else:
            print("Train without pretrained model...")
        epoch_str = 1
        global_step = 0

    # freeze all other layers except speaker embedding
    for p in net_g.parameters():
        p.requires_grad = True
    for p in net_d.parameters():
        p.requires_grad = True
    # for p in net_d.parameters():
    #     p.requires_grad = False
    # net_g.emb_g.weight.requires_grad = True
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    # optim_d = None
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()

def train_btn(dataset_path, dataset_name, continue_train, max_epochs, whisper_model_size, batch_size):
    output_log = ""
    lang2token = {
        'zh': "[ZH]",
        'ja': "[JA]",
    }
    if not torch.cuda.is_available():
        yield "抱歉无法训练，未检测到GPU"
    if not os.path.exists(dataset_path):
        yield "{} 输入错误，目录不存在，请检查。".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    else:
        wav_names = os.listdir(dataset_path)
        for wav_name in wav_names:
            if wav_name[-4:] != ".wav":
                return "{} 音频文件必须是wav格式的，该目录下存在非wav后缀的文件，请检查。".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        wav_paths = glob.glob(os.path.join(dataset_path, "*.wav"))
        for idx, wav_path in enumerate(wav_paths):
            rename_wav_path = os.path.join(dataset_path, "{}_{}.wav.tmp".format(dataset_name, idx + 1))
            shutil.move(wav_path, rename_wav_path)
        wav_paths = glob.glob(os.path.join(dataset_path, "*.wav.tmp"))
        for wav_path in wav_paths:
            shutil.move(wav_path, wav_path[:-4])
        output_log = "{} 【已完成】音频文件命名修改\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        yield output_log


        raw_audio_dir = "raw_audio"
        denoise_audio_dir = "denoised_audio"
        raw_audio_filelist = glob.glob(os.path.join(raw_audio_dir, "*.wav"))
        raw_audio_filelist = sorted(raw_audio_filelist, key = lambda x: int(x.split("_")[-1].split(".")[0]))

        with open(os.path.join("configs", "finetune_speaker.json"), 'r', encoding = 'utf-8') as f:
            hps = json.load(f)
        target_sr = hps['data']['sampling_rate']
        for file in raw_audio_filelist:
            if file.endswith(".wav"):
                os.system(f"demucs --two-stems=vocals {file}")
                output_log += "{} 【已完成】文件({})音频分离处理\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file.split("\\")[-1])
                yield output_log

        for file in raw_audio_filelist:
            fname = file.split("\\")[-1].replace(".wav", "")
            wav, sr = torchaudio.load(os.path.join("separated", "htdemucs", fname, "vocals.wav"), frame_offset=0, num_frames=-1, normalize=True,
                                    channels_first=True)
            # merge two channels into one
            wav = wav.mean(dim = 0).unsqueeze(0)
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
            torchaudio.save(os.path.join("denoised_audio", fname + ".wav"), wav, target_sr, channels_first=True)
            output_log += "{} 【已完成】文件({})音频采样处理\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file.split("\\")[-1])
            yield output_log

        denoise_audio_filelist = glob.glob(os.path.join(denoise_audio_dir, "*"))
        denoise_audio_filelist = sorted(denoise_audio_filelist, key = lambda x: int(x.split("_")[-1].split(".")[0]))

        model = whisper.load_model(whisper_model_size, download_root = ".\\whisper_model")
        speaker_annos = []
        for file in denoise_audio_filelist:
            result = model.transcribe(file)
            lang = result['language']
            if result['language'] not in list(lang2token.keys()):
                output_log += "{} 【报错】{}不支持\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), lang)
                yield output_log
                continue
            # segment audio based on segment results
            fname = file.split("\\")[-1]
            character_name = fname.rstrip(".wav").split("_")[0]
            code = fname.rstrip(".wav").split("_")[1]
            if not os.path.exists(os.path.join("segmented_character_voice", character_name)):
                os.mkdir(os.path.join("segmented_character_voice", character_name))
            wav, sr = torchaudio.load(file, frame_offset = 0, num_frames = -1, normalize = True,
                                    channels_first = True)
            
            for i, seg in enumerate(result['segments']):
                start_time = seg['start']
                end_time = seg['end']
                text = seg['text']
                text = lang2token[lang] + text.replace("\n", "") + lang2token[lang]
                text = text + "\n"
                wav_seg = wav[:, int(start_time*sr):int(end_time*sr)]
                wav_seg_name = f"{character_name}_{code}_{i}.wav"
                savepth = os.path.join("segmented_character_voice", character_name, wav_seg_name)
                speaker_annos.append(savepth + "|" + character_name + "|" + text)
                torchaudio.save(savepth, wav_seg, target_sr, channels_first=True)

            output_log += "{} 【已完成】文件({})音频识别\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), fname)
            yield output_log

        if len(speaker_annos) == 0:
            output_log += "{} 音频文件识别失败\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        with open("long_character_anno.txt", 'w', encoding='utf-8') as f:
            for line in speaker_annos:
                f.write(line)

        output_log += "{} 【已完成】音频对应txt文本生成\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        yield output_log

        # 数据预处理
        new_annos = []
        if os.path.exists("long_character_anno.txt"):
            with open("long_character_anno.txt", 'r', encoding='utf-8') as f:
                long_character_anno = f.readlines()
                new_annos += long_character_anno

        # Get all speaker names
        speakers = []
        for line in new_annos:
            path, speaker, text = line.split("|")
            if speaker not in speakers:
                speakers.append(speaker)

        with open(os.path.join("configs", "finetune_speaker.json"), 'r', encoding='utf-8') as f:
            hps = json.load(f)

        # assign ids to new speakers
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i
        # modify n_speakers
        hps['data']["n_speakers"] = len(speakers)
        # overwrite speaker names
        hps['speakers'] = speaker2id
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = batch_size
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"
        # save modified config
        with open(os.path.join("configs", "modified_finetune_speaker.json"), 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # STEP 2: clean annotations, replace speaker names with assigned speaker IDs
        import text

        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners']).replace("[ZH]", "")
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

        final_annos = cleaned_new_annos
        # save annotation file
        with open("final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)
        # save annotation file for validation
        with open("final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)

        output_log += "{} 【已完成】数据预处理，训练集、验证集切分\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        yield output_log

        output_log += "{} 【进行中】开始训练，训练进度请看后台\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        yield output_log

        train_main(continue_train, max_epochs)
    

if __name__ == "__main__":
    app = gr.Blocks()
    with app:
        with gr.Row():
            gr.HTML('<center style="font-size: 20px;"><b>VITS 模型训练</b></center>')
        with gr.Row():
            gr.HTML('<p style="font-size: 15px; text-align: left;">作者：Jack Cui</p>')
        with gr.Row():
            gr.HTML('<p style="font-size: 15px; text-align: left;">使用说明：<a href="https://space.bilibili.com/331507846", target="_black">https://space.bilibili.com/331507846</a></ps>')
        with gr.Row():
            gr.HTML('<p style="font-size: 15px; text-align: left;"><b style="color: red;">特别声明：</b>本项目仅限于学习交流，请勿用于非法用途，请勿使用非授权数据集进行训练。</p>')

        with gr.Row():
            with gr.Column():
                dataset_path = gr.Textbox(
                    label = "训练数据地址",
                    info = "wav音频文件，建议填写绝对路径",
                    lines = 1,
                    placeholder = "F:\\Code\\VITS_fast_finetune\\raw_audio",
                )
                dataset_name = gr.Textbox(
                    label = "模型名（角色名）",
                    info = "声音模型训练保存的名字，随便起",
                    lines = 1,
                    placeholder = "jackcui_test",
                )
                continue_train = gr.Radio(["是", "否"], value = "是", label = "是否重新训练", info = "重新训练选择是，接着已经保存的模型继续训练选择否")
                whisper_model_size = gr.Radio(["tiny", "base", "small", "medium", "large"], value = "medium", label = "语音识别模型", info = "8G显存选medium，8G以上选large")
                max_epochs = gr.Slider(2, 1000, value = 200, label = "训练epochs次数", info = "迭代训练的轮次，默认200")
                batch_size = gr.Slider(2, 256, step = 2, value = 4, label = "batch_size大小", info = "越大训练越快，显存消耗越大")

            with gr.Column():
                text_output = gr.TextArea(
                    label = "输出结果",
                    lines = 23,
                    )
                btn = gr.Button("开始训练")
                btn.click(train_btn,
                            inputs = [dataset_path, dataset_name, continue_train, max_epochs, whisper_model_size, batch_size],
                            outputs = text_output)            
                
    webbrowser.open("http://127.0.0.1:8088")
    app.queue(concurrency_count=5, max_size=20).launch(server_name="127.0.0.1", server_port=8088)
