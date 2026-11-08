"""Fetch the author's public Drive archive with resumable, validated ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


URL = ('https://drive.usercontent.google.com/download?'
       'id=1fyNCZ62pEItTQYEBzLwsZ9ehX_-Ai3qT&export=download&confirm=t')
ARCHIVE_BYTES = 569780725
CHUNK_BYTES = 8 * 1024 * 1024
DESTINATION = Path(__file__).resolve().parent.parent / 'downloads' / 'pretrain.tar.gz'


def fetch_range(start):
    end = min(start + CHUNK_BYTES, ARCHIVE_BYTES) - 1
    path = DESTINATION.parent / 'pretrain-ranges' / f'{start:010}.part'
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    request = Request(f'{URL}&segment={start}', headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=75) as response:
                expected = f'bytes {start}-{end}/{ARCHIVE_BYTES}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise OSError(f'Invalid HTTP range for {start}: {response.status}')
                data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise OSError(f'Incomplete range at {start}')
            path.write_bytes(data)
            return path
        except OSError:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


if __name__ == '__main__':
    if not DESTINATION.exists():
        (DESTINATION.parent / 'pretrain-ranges').mkdir(parents=True, exist_ok=True)
        assembled = DESTINATION.with_suffix('.assembling')
        with ThreadPoolExecutor(max_workers=10) as pool, assembled.open('wb') as output:
            for segment in pool.map(fetch_range, range(0, ARCHIVE_BYTES, CHUNK_BYTES)):
                with segment.open('rb') as source:
                    shutil.copyfileobj(source, output)
        if assembled.stat().st_size != ARCHIVE_BYTES:
            raise OSError('Invalid assembled archive size')
        assembled.replace(DESTINATION)
    with DESTINATION.open('rb') as archive:
        print(f'{hashlib.file_digest(archive, "sha256").hexdigest()}  {DESTINATION}', flush=True)
