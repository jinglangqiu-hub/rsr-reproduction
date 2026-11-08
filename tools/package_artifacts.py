"""Archive existing experiment outputs without loading a model or training."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs' / 'artifact-manifest.json'
ARCHIVE = ROOT / 'artifacts' / 'rsr-experiment-artifacts-v1.zip'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_files():
    return sorted(path for path in (ROOT / 'runs').rglob('*')
                  if path.is_file() and path.suffix not in ('.npy', '.lock', '.tmp', '.part'))


def build():
    if ARCHIVE.exists():
        raise SystemExit(f'Refusing to overwrite existing archive: {ARCHIVE}')
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(ARCHIVE, 'x', allowZip64=True) as archive:
        for path in selected_files():
            relative = path.relative_to(ROOT).as_posix()
            records.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest(path)})
            compression = zipfile.ZIP_STORED if path.suffix == '.npz' else zipfile.ZIP_DEFLATED
            archive.write(path, relative, compress_type=compression, compresslevel=6)
    manifest = {
        'snapshot': 'v0.1.0-experiments',
        'archive': ARCHIVE.name,
        'archive_bytes': ARCHIVE.stat().st_size,
        'archive_sha256': digest(ARCHIVE),
        'file_count': len(records),
        'uncompressed_bytes': sum(item['bytes'] for item in records),
        'selection': 'runs/** except .npy, .lock, .tmp, .part',
        'excluded': ['large generated .npy arrays', 'author source data and public pretrained arrays'],
        'files': records,
    }
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'files'}, indent=2))


def verify(path):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if path.stat().st_size != manifest['archive_bytes'] or digest(path) != manifest['archive_sha256']:
        raise SystemExit('Archive size or SHA-256 does not match the published manifest')
    expected = {item['path']: item for item in manifest['files']}
    if len(expected) != manifest['file_count']:
        raise SystemExit('Duplicate or inconsistent manifest entries')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise SystemExit('Archive entries differ from the published manifest')
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if (name.is_absolute() or '..' in name.parts or name.parts[0] != 'runs'
                    or '\\' in member.filename or ':' in member.filename):
                raise SystemExit(f'Unsafe archive member: {member.filename}')
            record = expected[member.filename]
            with archive.open(member) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if member.file_size != record['bytes'] or actual != record['sha256']:
                raise SystemExit(f'Content mismatch: {member.filename}')
    print(f'Verified archive SHA-256 and all {len(expected)} member hashes; no files extracted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build', action='store_true')
    mode.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.build:
        build()
    else:
        verify(args.verify.resolve())


if __name__ == '__main__':
    main()
