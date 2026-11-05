"""Expose Edge TTS as a loopback OpenAI-compatible speech endpoint for AIRI."""
import asyncio
import json
import math
import re
import time

import edge_tts
from aiohttp import web

ORIGINS = {'http://127.0.0.1:5173', 'http://localhost:5173'}
VOICES = {'zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural'}
SLOTS = asyncio.Semaphore(3)

@web.middleware
async def cors(request, handler):
    origin = request.headers.get('Origin')
    if origin and origin not in ORIGINS:
        return web.json_response({'error': {'message': 'Origin not allowed'}}, status=403)
    if request.method == 'OPTIONS':
        response = web.Response(status=204)
    else:
        response = await handler(request)
    if origin:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Cache-Control'] = 'no-store'
    return response

async def health(request):
    return web.json_response({'service': 'airi-edge-tts', 'status': 'ok', 'voice': 'zh-CN-XiaoxiaoNeural'})

async def models(request):
    return web.json_response({'object': 'list', 'data': [{'id': 'edge-tts', 'object': 'model', 'created': 0, 'owned_by': 'local'}]})

async def speech(request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('Expected a JSON object')
        text = body.get('input')
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 8000:
            raise ValueError('input must contain 1 to 8000 characters')
        # AIRI control markers are not spoken even if they reach the TTS boundary.
        text = re.sub(r'<\|.*?\|>', '', text).strip()
        if not text:
            raise ValueError('No spoken text')
        voice = body.get('voice') or 'zh-CN-XiaoxiaoNeural'
        # OpenAI-compatible clients default to alloy before the custom voice is selected.
        if voice == 'alloy':
            voice = 'zh-CN-XiaoxiaoNeural'
        if voice not in VOICES:
            raise ValueError('Choose zh-CN-XiaoxiaoNeural or zh-CN-XiaoyiNeural')
        if body.get('response_format', 'mp3') != 'mp3':
            raise ValueError('This endpoint returns mp3 audio')
        speed = float(body.get('speed', 1.0))
        if not math.isfinite(speed) or not .5 <= speed <= 2:
            raise ValueError('speed must be between 0.5 and 2')
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return web.json_response({'error': {'message': str(exc)}}, status=400)
    started = time.monotonic()
    try:
        async with SLOTS:
            async with asyncio.timeout(45):
                communicator = edge_tts.Communicate(text, voice, rate=f'{round((speed - 1) * 100):+d}%', pitch='+0Hz')
                chunks = [chunk['data'] async for chunk in communicator.stream() if chunk['type'] == 'audio']
                audio = b''.join(chunks)
                if not audio:
                    raise RuntimeError('No audio returned')
        # Record performance only, never conversation text or API credentials.
        print(f'speech ok voice={voice} bytes={len(audio)} seconds={time.monotonic()-started:.2f}', flush=True)
        return web.Response(body=audio, content_type='audio/mpeg')
    except Exception as exc:
        print(f'speech failed type={type(exc).__name__}', flush=True)
        return web.json_response({'error': {'message': f'Speech service unavailable ({type(exc).__name__})'}}, status=502)

app = web.Application(middlewares=[cors], client_max_size=65536)
app.router.add_get('/health', health)
app.router.add_get('/v1/models', models)
app.router.add_post('/v1/audio/speech', speech)
app.router.add_route('OPTIONS', '/{path:.*}', health)
if __name__ == '__main__':
    web.run_app(app, host='127.0.0.1', port=8008, access_log=None)
