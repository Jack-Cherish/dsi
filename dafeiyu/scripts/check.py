"""Check release files, nested model references, and accidental private data."""
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = re.compile(rb'[A-Za-z]:[\\/]Users[\\/][^\s"<>]+', re.I)
SECRET = re.compile(rb'\bsk-[A-Za-z0-9_-]{24,}')


def scan(data, label):
    # Scan binaries too, and descend into model packages.
    for encoding in ('utf-8', 'utf-16-le'):
        decoded = data.decode(encoding, errors='ignore').encode('utf-8')
        if PRIVATE_PATH.search(decoded) or SECRET.search(decoded):
            raise ValueError('Possible private data in ' + label + ' (value not printed)')
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name:
                    raise ValueError('Unsafe archive member: ' + label)
                if not name.endswith('/'):
                    scan(archive.read(name), label + ':' + name)


def check_model():
    with zipfile.ZipFile(ROOT / 'assets/character.airi.zip') as card_zip:
        manifest = json.loads(card_zip.read('manifest.json'))
        card = json.loads(card_zip.read('card.json'))
        assert card['spec'] == 'chara_card_v3'
        assert '【明确动作请求的执行规则】' in card['data']['system_prompt']
        model_path = manifest['resources']['displayModel']['path']
        with zipfile.ZipFile(io.BytesIO(card_zip.read(model_path))) as model_zip:
            names = model_zip.namelist()
            model_name = next(name for name in names if name.endswith('.model3.json'))
            model = json.loads(model_zip.read(model_name))
            parent = PurePosixPath(model_name).parent
            refs = model['FileReferences']
            targets = [refs['Moc'], *refs['Textures']]
            for field in ('Physics', 'Pose', 'DisplayInfo'):
                if refs.get(field):
                    targets.append(refs[field])
            targets.extend(item['File'] for item in refs.get('Expressions', []))
            targets.extend(item['File'] for motions in refs.get('Motions', {}).values() for item in motions)
            for target in targets:
                assert str(parent / target) in names, 'Missing model resource: ' + target
            for action in ('wave', 'earTwitch', 'tailSwing'):
                assert any(name.endswith('.' + action + '.motion3.json') for name in names), action
            assert model_zip.read(str(parent / refs['Moc'])).startswith(b'MOC3')


def main():
    sums = ROOT / 'SHA256SUMS'
    for line in sums.read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()), 'Checksum path leaves root'
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, 'Checksum mismatch: ' + name
        assert not any(part in {'.runtime', 'node_modules', '__pycache__', 'recordings'} for part in Path(name).parts)
        assert path.name != 'voice.local.json'
        scan(data, name)
        if name.endswith('.py'):
            ast.parse(data, filename=name)
    upstream = json.loads((ROOT / 'upstream.json').read_text())
    for item in upstream['overlay']:
        assert hashlib.sha256((ROOT / 'overlay' / item['path']).read_bytes()).hexdigest() == item['sha256']
    check_model()
    print('PASS: checksums, Python syntax, source overlay, model resources and private-data scan.')


if __name__ == '__main__':
    main()
