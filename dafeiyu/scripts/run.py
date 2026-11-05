"""Start only this installation's local services and AIRI application."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'


def probe(port, path, marker):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3) as response:
            return marker in response.read(200000)
    except OSError:
        return False


def port_busy(port):
    with socket.socket() as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0


def process_module():
    # The parent can be any Python installation. Query processes with the private venv.
    sys.path.insert(0, str(RUNTIME / 'tts-env/Lib/site-packages'))
    import psutil
    return psutil


def owned(name):
    record = RUNTIME / f'{name}.process.json'
    if not record.exists():
        return None
    psutil = process_module()
    try:
        data = json.loads(record.read_text())
        process = psutil.Process(data['pid'])
        if abs(process.create_time() - data['created']) > 0.01:
            return None
        if process.cmdline() != data['command']:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, KeyError):
        return None


def spawn(name, command, cwd, port=None, path='/health', marker=b'', timeout=60):
    existing = owned(name)
    if existing:
        if port is None:
            print(name + ' is already running.')
            return
    elif port is not None and port_busy(port):
        raise RuntimeError(f'Port {port} belongs to another service. No process was stopped. Close that service before retrying.')
    if not existing:
        environment = os.environ.copy()
        environment.pop('ELECTRON_RUN_AS_NODE', None)
        environment['APP_USER_DATA_PATH'] = str(RUNTIME / 'desktop-profile')
        environment['APP_DEBUG'] = '0'
        with (RUNTIME / f'{name}.log').open('ab') as log:
            child = subprocess.Popen([str(p) for p in command], cwd=cwd, env=environment,
                                     stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        existing = process_module().Process(child.pid)
        (RUNTIME / f'{name}.process.json').write_text(json.dumps({
            'pid': existing.pid, 'created': existing.create_time(), 'command': existing.cmdline()}))
    if port is None:
        time.sleep(3)
        if not existing.is_running():
            raise RuntimeError(f'{name} exited. Check .runtime/{name}.log')
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe(port, path, marker):
            return
        if not existing.is_running():
            break
        time.sleep(0.5)
    raise RuntimeError(f'{name} did not become ready. Check .runtime/{name}.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['desktop', 'web', 'stop', 'status'], nargs='?', default='desktop')
    args = parser.parse_args()
    state = RUNTIME / 'installation.json'
    if not state.exists():
        raise SystemExit('Run install.ps1 and complete AGENTS.md first.')
    if args.mode in ('stop', 'status'):
        for name in ('desktop', 'web', 'gptsovits-bridge', 'gptsovits-api', 'edge-tts'):
            process = owned(name)
            print(name + (': running' if process else ': stopped'))
            if process and args.mode == 'stop':
                psutil = process_module()
                children = process.children(recursive=True)
                for child in reversed(children):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                _, alive = psutil.wait_procs(children + [process], timeout=5)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
        return
    if json.loads(state.read_text())['voice'] == 'gptsovits':
        voice = ROOT / 'voice.local.json'
        if not voice.exists():
            raise RuntimeError('Create voice.local.json from voice.example.json with your own recording.')
        config = json.loads(voice.read_text(encoding='utf-8-sig'))
        reference = Path(config.get('ref_audio_path', ''))
        if not reference.is_absolute():
            reference = ROOT / reference
        if not reference.is_file() or not config.get('prompt_text', '').strip():
            raise RuntimeError('Reference recording or its transcript is missing.')
        py = RUNTIME / 'voice-env/Scripts/python.exe'
        spawn('gptsovits-api', [py, ROOT / 'scripts/run-voice.py'], ROOT,
              9880, '/openapi.json', b'"/tts"', 300)
        spawn('gptsovits-bridge', [py, ROOT / 'scripts/gptsovits-bridge.py'], ROOT,
              8009, '/health', b'airi-gptsovits')
    else:
        spawn('edge-tts', [RUNTIME / 'tts-env/Scripts/python.exe', ROOT / 'scripts/edge-tts-server.py'], ROOT,
              8008, '/health', b'airi-edge-tts')
    repo = RUNTIME / 'airi'
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js 24 is missing from PATH.')
    webapp = repo / 'apps/stage-web'
    if not (webapp / 'dist/index.html').exists():
        raise RuntimeError('Web build is missing. Run install.ps1.')
    spawn('web', [node, webapp / 'node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1',
                  '--port', '5173', '--strictPort'], webapp, 5173, '/', b'AIRI')
    if args.mode == 'web':
        webbrowser.open('http://127.0.0.1:5173/')
    else:
        app = repo / 'apps/stage-tamagotchi'
        electron = app / 'node_modules/electron/dist/electron.exe'
        if not electron.exists() or not (app / 'out/launcher/index.mjs').exists():
            raise RuntimeError('Desktop build is missing. Run install.ps1.')
        spawn('desktop', [electron, app], app)
    print('AIRI is ready.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
