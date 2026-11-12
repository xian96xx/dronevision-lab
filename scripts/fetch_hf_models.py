import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'work'/'hf-cache')
os.environ['HF_HUB_DISABLE_XET']='1'
import hashlib
import json
from huggingface_hub import HfApi, snapshot_download

def main():
    api=HfApi()
    for repo,folder in [('facebook/sam2.1-hiera-tiny','sam2.1-tiny'),('nvidia/mit-b0','mit-b0')]:
        info=api.model_info(repo)
        files=[x.rfilename for x in info.siblings]
        weights=['model.safetensors'] if 'model.safetensors' in files else ['pytorch_model.bin']
        selected=weights+[x for x in files if x in ['config.json','preprocessor_config.json','processor_config.json','README.md'] or x.startswith('LICENSE')]
        dest=ROOT/'models'/folder
        snapshot_download(repo,revision=info.sha,local_dir=dest,allow_patterns=selected,max_workers=2)
        records={str(p.relative_to(dest)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in dest.iterdir() if p.is_file()}
        (ROOT/'reports'/f'{folder}-source.json').write_text(json.dumps({'repository':repo,'revision':info.sha,'sha256':records},indent=2))
        print(repo,info.sha,records,flush=True)

if __name__=='__main__':main()
