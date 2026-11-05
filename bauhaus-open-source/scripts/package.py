"""Package an already built project, with no third-party Python dependencies."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent.parent
archive = root.parent / '包豪斯样板间-HTML源码与Skills.zip'
excluded = {'node_modules', '.git', 'release-public', '__pycache__', '.DS_Store', 'Thumbs.db'}
files = sorted(p for p in root.rglob('*') if p.is_file()
               and not any(part in excluded for part in p.relative_to(root).parts)
               and p.name != 'MANIFEST.json' and p.suffix != '.log')
for required in ['standalone.html', 'dist/index.html', 'dist/app.js', 'README.md', 'LICENSE']:
    if not (root / required).is_file():
        raise SystemExit(f'Missing {required}; run npm run build first.')
manifest = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(root / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as package:
    for p in files + [root / 'MANIFEST.json']:
        package.write(p, 'bauhaus-open-source/' + p.relative_to(root).as_posix())
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
    for record in manifest:
        data = package.read('bauhaus-open-source/' + record['path'])
        assert len(data) == record['bytes']
        assert hashlib.sha256(data).hexdigest() == record['sha256']
print(json.dumps({'archive': str(archive), 'files': len(files)+1,
                  'bytes': archive.stat().st_size, 'crcAndSha256': 'PASS'}, ensure_ascii=False))
