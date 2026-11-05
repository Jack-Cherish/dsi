"""Prepare the pinned AIRI source and build a private Windows installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'


def run(args, cwd=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path leaves the installation directory')
    return path


def checkout(repository, commit, destination):
    if not (destination / '.git').exists():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f'Refusing to replace existing directory: {destination}')
        destination.mkdir(parents=True, exist_ok=True)
        run(['git', 'init', destination])
        run(['git', 'config', 'core.autocrlf', 'false'], destination)
        run(['git', 'remote', 'add', 'origin', repository], destination)
        run(['git', 'fetch', '--depth', '1', 'origin', commit], destination)
        run(['git', 'checkout', '--detach', 'FETCH_HEAD'], destination)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip()
    if actual != commit:
        raise RuntimeError('Source version differs. Use a new installation directory.')


def apply_overlay(repo, manifest):
    pending = []
    # Check all files before changing any file. Preserve later user edits.
    for item in manifest['overlay']:
        source = inside(ROOT / 'overlay', item['path'])
        target = inside(repo, item['path'])
        if digest(source) != item['sha256']:
            raise RuntimeError(f'Package checksum failed: {item["path"]}')
        current = digest(target) if target.exists() else None
        if current == item['sha256']:
            continue
        if current != item['base_sha256']:
            raise RuntimeError(f'Local changes detected: {item["path"]}. No files overwritten.')
        pending.append((source, target))
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def make_card(voice):
    target = RUNTIME / 'dafeiyu.airi.zip'
    with zipfile.ZipFile(ROOT / 'assets/character.airi.zip') as source:
        card = json.loads(source.read('card.json'))
        speech = card['data']['extensions']['airi']['modules']['speech']
        speech.update(provider='openai-compatible-audio-speech',
                      model='gpt-sovits' if voice == 'gptsovits' else 'edge-tts',
                      voice_id='fish-girl' if voice == 'gptsovits' else 'zh-CN-XiaoxiaoNeural')
        # Let each recipient choose a model actually offered by their provider.
        card['data']['extensions']['airi']['modules']['consciousness'] = {'provider': '', 'model': ''}
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as output:
            for name in source.namelist():
                content = json.dumps(card, ensure_ascii=False, indent=2).encode() if name == 'card.json' else source.read(name)
                output.writestr(name, content)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='Fetch and check source without installing dependencies')
    parser.add_argument('--voice', choices=['edge', 'gptsovits'], default='edge')
    args = parser.parse_args()
    if sys.platform != 'win32':
        raise SystemExit('This installer supports Windows x64. Other platforms need separate validation.')
    manifest = json.loads((ROOT / 'upstream.json').read_text())
    if not shutil.which('git'):
        raise SystemExit('Install Git first. See AGENTS.md.')
    RUNTIME.mkdir(exist_ok=True)
    lock = RUNTIME / 'install.lock'
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise SystemExit('Another installation holds .runtime/install.lock. Check it before retrying.')
    try:
        repo = RUNTIME / 'airi'
        checkout(manifest['repository'], manifest['commit'], repo)
        apply_overlay(repo, manifest)
        make_card(args.voice)
        if args.prepare_only:
            print('Pinned source and character package are ready.')
            return
        node = shutil.which('node')
        npx = shutil.which('npx')
        uv = shutil.which('uv')
        if not node or not npx or not uv:
            raise RuntimeError('Install Node.js 24 and uv first. See AGENTS.md.')
        if not subprocess.check_output([node, '--version'], text=True).startswith('v24.'):
            raise RuntimeError('Use Node.js 24 LTS for this pinned AIRI version.')
        pnpm = [npx, '--yes', manifest['package_manager']]
        run(pnpm + ['install', '--frozen-lockfile'], repo)
        run(pnpm + ['-F', '@proj-airi/stage-web', 'build'], repo)
        run(pnpm + ['-F', '@proj-airi/stage-tamagotchi', 'build'], repo)
        run(pnpm + ['-F', '@proj-airi/stage-tamagotchi', 'exec', 'install-electron'], repo)
        env = RUNTIME / 'tts-env'
        if not (env / 'Scripts/python.exe').exists():
            run([uv, 'venv', '--python', '3.11', env])
        run([uv, 'pip', 'install', '--python', env / 'Scripts/python.exe', '-r', ROOT / 'scripts/tts-requirements.txt'])
        if args.voice == 'gptsovits':
            run([sys.executable, ROOT / 'scripts/install-voice.py'])
        (RUNTIME / 'installation.json').write_text(json.dumps({'commit': manifest['commit'], 'voice': args.voice}, indent=2))
        print('Build complete. Agent: finish the first-run settings in AGENTS.md.')
        print('Import character package: ' + str(RUNTIME / 'dafeiyu.airi.zip'))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
