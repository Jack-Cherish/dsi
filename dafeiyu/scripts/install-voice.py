"""Install the pinned CPU GPT-SoVITS backend without distributing recordings."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from install import ROOT, RUNTIME, checkout, inside, run


def download(item):
    target = inside(RUNTIME, item['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == item['sha256']:
        return
    partial = target.with_name(target.name + '.partial')
    print('Downloading ' + target.name, flush=True)
    with urllib.request.urlopen(item['url'], timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output, 1024 * 1024)
    if sha256(partial) != item['sha256']:
        raise RuntimeError('Download checksum mismatch: ' + target.name)
    partial.replace(target)


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    uv = shutil.which('uv')
    if not uv or not shutil.which('ffmpeg'):
        raise RuntimeError('GPT-SoVITS needs uv and FFmpeg in PATH. See AGENTS.md.')
    manifest = json.loads((ROOT / 'voice-models.json').read_text())
    repo = RUNTIME / 'GPT-SoVITS'
    checkout(manifest['repository'], manifest['commit'], repo)
    environment = RUNTIME / 'voice-env'
    py = environment / 'Scripts/python.exe'
    if not py.exists():
        run([uv, 'venv', '--python', '3.11', environment])
    # Install CPU wheels from their publisher before resolving the PyPI packages.
    run([uv, 'pip', 'install', '--python', py, '--index-url', 'https://download.pytorch.org/whl/cpu',
         'torch==2.7.1+cpu', 'torchaudio==2.7.1+cpu'])
    run([uv, 'pip', 'install', '--python', py, '-r', ROOT / 'scripts/voice-requirements.txt'])
    for item in manifest['downloads']:
        download(item)
        if 'extract_to' in item:
            target = inside(RUNTIME, item['extract_to'])
            with zipfile.ZipFile(inside(RUNTIME, item['path'])) as archive:
                for name in archive.namelist():
                    inside(target, name)
                archive.extractall(target)
    print('GPT-SoVITS installed. Supply voice.local.json and your own reference recording.')


if __name__ == '__main__':
    main()
