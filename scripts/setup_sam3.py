"""Explicit one-time download after the user has obtained facebook/sam3 access."""
from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download


def main():
    target=Path.home()/'.cache/cad-rendering/sam3'
    # Standard locally stored HF authentication is used, never printed or copied into layouts.
    try:
        info=HfApi().model_info('facebook/sam3')
        snapshot_download('facebook/sam3',revision=info.sha,local_dir=target,
                          allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
        (target/'source_revision.json').write_text(json.dumps({'repository':'facebook/sam3','revision':info.sha}))
    except Exception:
        raise SystemExit('SAM 3 download unavailable. Obtain access at https://huggingface.co/facebook/sam3 and authenticate locally with hf auth login, then retry. No paid inference was requested.')
    print(f'SAM 3 cached locally: {target}')


if __name__=='__main__': main()
