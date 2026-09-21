"""One-time official SAM ViT-B checkpoint download; never invoked by the UI."""
from hashlib import sha256
from pathlib import Path
import urllib.request

URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
EXPECTED = 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'


def main():
    target = Path.home() / '.cache/cad-rendering/sam/sam_vit_b_01ec64.pth'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target.read_bytes()).hexdigest() == EXPECTED:
        print(f'Checkpoint already cached: {target}')
        return
    temporary = target.with_suffix('.download')
    urllib.request.urlretrieve(URL, temporary)
    if sha256(temporary.read_bytes()).hexdigest() != EXPECTED:
        temporary.unlink(missing_ok=True)
        raise RuntimeError('Checkpoint checksum mismatch; existing model was not replaced.')
    temporary.replace(target)
    print(f'Checkpoint ready: {target}')


if __name__ == '__main__':
    main()
