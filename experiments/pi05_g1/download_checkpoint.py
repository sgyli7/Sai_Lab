"""Download official PI05 base with GCS MD5 or composite-object CRC32C checks."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    prefix = 'checkpoints/pi05_base/'
    url = 'https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=' + quote(prefix, safe='') + '&maxResults=1000'
    with urlopen(url, timeout=30) as response:
        listing = json.load(response)
    if listing.get('nextPageToken'):
        raise RuntimeError('Unexpected paginated base checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gcs_manifest.json').write_text(json.dumps(listing, indent=2))

    def fetch(item):
        name, size = item['name'], int(item['size'])
        target = args.output / name.removeprefix(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('md5Hash') or item.get('crc32c')
        if not expected:
            raise ValueError(f'Missing GCS checksum: {name}')
        def valid():
            if not target.is_file() or target.stat().st_size != size:
                return False
            with target.open('rb') as f:
                if item.get('md5Hash'):
                    checksum = hashlib.file_digest(f, 'md5').digest()
                else:
                    import google_crc32c
                    crc = google_crc32c.Checksum()
                    while block := f.read(8 * 1024 * 1024):
                        crc.update(block)
                    checksum = crc.digest()
                return base64.b64encode(checksum).decode() == expected
        if valid():
            print('CACHED', name, flush=True)
            return
        temp = target.with_name(target.name + '.partial')
        for attempt in range(3):
            try:
                offset = temp.stat().st_size if temp.exists() else 0
                if offset > size:
                    temp.unlink(); offset = 0
                req = Request('https://storage.googleapis.com/openpi-assets/' + quote(name, safe='/') + '?generation=' + item['generation'],
                              headers={'Range': f'bytes={offset}-'} if offset else {})
                if offset == size:
                    if not temp.exists():
                        temp.touch()
                    temp.replace(target)
                else:
                    with urlopen(req, timeout=90) as response:
                        append = offset > 0 and response.status == 206
                        with temp.open('ab' if append else 'wb') as stream:
                            while chunk := response.read(8 * 1024 * 1024):
                                stream.write(chunk)
                    temp.replace(target)
                if not valid():
                    target.unlink(missing_ok=True)
                    raise ValueError('GCS size/checksum mismatch')
                print('DOWNLOADED', name, size, flush=True)
                return
            except Exception as exc:
                print('RETRY', name, attempt, str(exc), flush=True)
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, listing['items']))
    (args.output / 'download_complete.json').write_text(json.dumps({
        'source': 'gs://openpi-assets/checkpoints/pi05_base', 'objects': len(listing['items']),
        'bytes': sum(int(i['size']) for i in listing['items']), 'all_gcs_checksums_verified': True,
        'completed_unix': time.time()}, indent=2))


if __name__ == '__main__':
    main()
