import modal
from types import SimpleNamespace

WEIGHTS = "/pretrained_weights"

app = modal.App(name="liveportrait")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(
        "requirements.txt"
    )
    .add_local_dir("assets", remote_path="/root/assets")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_python_source("src")
)

with image.imports():
    import os
    import os.path as osp
    import subprocess
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

volume = modal.Volume.from_name("lp-weights", create_if_missing=True)

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

def read_full_video(file_path):
    with open(file_path, "rb") as f:
        return f.read()  # Reads ENTIRE file into memory! No suitable for very large files

@app.cls(
    image=image,
    gpu="A10G",
    volumes={WEIGHTS: volume},
    timeout=1000 # in seconds
)

class Process:
    @modal.enter()
    def enter(self):
        import os
        from huggingface_hub import snapshot_download

        # Download checkpoint if not exists
        ckpt_path = WEIGHTS + "/README.md"

        if not os.path.exists(ckpt_path):
            print("Downloading checkpoint files.")

            snapshot_download(
                "KwaiVGI/LivePortrait",
                local_dir=WEIGHTS
            )

    @modal.method()
    def execute_pipeline(self, args):

        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

        if not fast_check_ffmpeg():
            raise ImportError(
                "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
            )

        fast_check_args(args)

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )

        # run
        wfp, wfp_concat = live_portrait_pipeline.execute(args)

        return read_full_video(wfp)

@app.local_entrypoint()
def main():
    arguments = {
        'source' : 'assets/examples/source/s9.jpg',
        'driving' : 'assets/examples/driving/d0.mp4',
        'output_dir' : '/tmp'
    }
    arguments = SimpleNamespace(**arguments)
    video_bytes = Process().execute_pipeline.remote(arguments)

    with open("output/output.mp4", "wb") as f:
        f.write(video_bytes)  # Write the bytes to a new file
