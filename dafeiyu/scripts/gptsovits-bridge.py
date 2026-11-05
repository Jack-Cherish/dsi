"""Loopback OpenAI-compatible speech adapter; voice settings stay on disk."""
import asyncio
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import shutil
import wave

from aiohttp import ClientSession, ClientTimeout, web

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
FFMPEG = shutil.which('ffmpeg')
if not FFMPEG:
    raise RuntimeError('FFmpeg is missing from PATH')
BACKEND = 'http://127.0.0.1:9880'
ORIGINS = {'http://127.0.0.1:5173', 'http://localhost:5173'}
SLOTS = asyncio.Semaphore(1)

def voice_config():
    config = json.loads((ROOT / 'voice.local.json').read_text(encoding='utf-8-sig'))
    reference = Path(config['ref_audio_path'])
    config['ref_audio_path'] = str(reference if reference.is_absolute() else ROOT / reference)
    return config

def voice_ready(config):
    ref = config.get('ref_audio_path', '')
    return bool(ref and Path(ref).is_file() and config.get('prompt_text', '').strip())

@web.middleware
async def cors(request, handler):
    origin = request.headers.get('Origin')
    if origin and origin not in ORIGINS:
        return web.json_response({'error': {'message': 'Origin not allowed'}}, status=403)
    response = web.Response(status=204) if request.method == 'OPTIONS' else await handler(request)
    if origin:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Cache-Control'] = 'no-store'
    return response

async def health(request):
    config = voice_config()
    backend_ready = False
    try:
        async with ClientSession(timeout=ClientTimeout(total=3)) as session:
            async with session.get(BACKEND + '/openapi.json') as response:
                backend_ready = response.status == 200 and '/tts' in (await response.json()).get('paths', {})
    except Exception:
        pass
    return web.json_response({'service': 'airi-gptsovits', 'backend_ready': backend_ready,
                             'voice_ready': voice_ready(config), 'voice': config['voice']})

async def models(request):
    return web.json_response({'object': 'list', 'data': [
        {'id': 'gpt-sovits', 'object': 'model', 'created': 0, 'owned_by': 'local'}]})

async def speech(request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('Expected a JSON object')
        text = body.get('input')
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError('input must contain 1 to 2000 characters')
        text = re.sub(r'<\|.*?\|>', '', text).strip()
        if not text:
            raise ValueError('No spoken text')
        config = voice_config()
        if body.get('voice', config['voice']) not in (config['voice'], 'alloy'):
            raise ValueError('Unknown configured voice')
        speed = float(body.get('speed', 1))
        if not math.isfinite(speed) or not .5 <= speed <= 2:
            raise ValueError('speed must be between 0.5 and 2')
        pace = float(config.get('playback_speed', 1))
        if not math.isfinite(pace) or not .5 <= pace <= 2:
            raise ValueError('playback_speed must be between 0.5 and 2')
        speed = min(2, max(.5, speed * pace))
        fmt = body.get('response_format', 'mp3')
        if fmt not in ('mp3', 'wav'):
            raise ValueError('Supported response formats: mp3, wav')
    except (ValueError, TypeError, KeyError) as exc:
        return web.json_response({'error': {'message': str(exc)}}, status=400)
    if not voice_ready(config):
        return web.json_response({'error': {'message': 'Reference audio and its transcript are not configured yet'}}, status=503)
    payload = dict(text=text, text_lang='zh', ref_audio_path=config['ref_audio_path'],
                   prompt_text=config['prompt_text'], prompt_lang=config.get('prompt_lang', 'zh'),
                   top_k=15, top_p=1, temperature=1, text_split_method='cut5', batch_size=1,
                   speed_factor=1, fragment_interval=.15, media_type='wav', streaming_mode=False, parallel_infer=True,
                   use_cuda_graph=False, repetition_penalty=1.35)
    try:
        async with SLOTS:
            async with ClientSession(timeout=ClientTimeout(total=300)) as session:
                # Do not forward AIRI's Authorization header to the local model.
                async with session.post(BACKEND + '/tts', json=payload) as response:
                    if response.status != 200:
                        raise RuntimeError(f'GPT-SoVITS HTTP {response.status}')
                    audio = await response.read()
                    if not audio.startswith(b'RIFF'):
                        raise RuntimeError('Backend did not return WAV audio')
            if fmt == 'mp3' or speed != 1:
                # Changing the model's speed disables its parallel decoder. Tempo processing
                # after synthesis keeps that path fast and preserves the voice's pitch.
                with wave.open(io.BytesIO(audio)) as source:
                    rate, channels = source.getframerate(), source.getnchannels()
                encoding = ['-f', 'mp3', '-codec:a', 'libmp3lame', '-q:a', '3'] if fmt == 'mp3' else ['-f', 's16le', '-codec:a', 'pcm_s16le']
                proc = await asyncio.create_subprocess_exec(
                    str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
                    '-af', f'atempo={speed}', *encoding, 'pipe:1',
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                try:
                    audio, _ = await asyncio.wait_for(proc.communicate(audio), timeout=30)
                except BaseException:
                    if proc.returncode is None:
                        proc.kill()
                    await proc.wait()
                    raise
                if proc.returncode or not audio:
                    raise RuntimeError('Audio conversion failed')
                if fmt == 'wav':
                    output = io.BytesIO()
                    with wave.open(output, 'wb') as wav:
                        wav.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
                        wav.writeframes(audio)
                    audio = output.getvalue()
        return web.Response(body=audio, content_type='audio/mpeg' if fmt == 'mp3' else 'audio/wav')
    except Exception as exc:
        print(f'speech failed: {type(exc).__name__}', flush=True)
        return web.json_response({'error': {'message': f'Speech unavailable ({type(exc).__name__}); check local backend log'}}, status=502)

app = web.Application(middlewares=[cors], client_max_size=16384)
app.router.add_get('/health', health)
app.router.add_get('/v1/models', models)
app.router.add_post('/v1/audio/speech', speech)
app.router.add_route('OPTIONS', '/{path:.*}', health)
if __name__ == '__main__':
    web.run_app(app, host='127.0.0.1', port=8009, access_log=None)
