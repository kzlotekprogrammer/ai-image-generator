"""Generate images locally on the GPU with Stable Diffusion XL (diffusers).

Examples:
    python generate.py "a lighthouse on a cliff at sunset, oil painting"
    python generate.py "cyberpunk cat" --preset turbo -n 4
    python generate.py --prompts-file prompts.txt --push
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import truststore

# Use the Windows certificate store (Norton intercepts HTTPS with its own root CA).
truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent
# Keep downloaded models inside the project (gitignored) instead of the user profile.
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

OUTPUTS = ROOT / "outputs"
GALLERY = ROOT / "GALLERY.md"

PRESETS = {
    # Best quality; ~30 s/image on an RTX 3070.
    "sdxl": {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "guidance": 6.5,
        "negative": "lowres, blurry, jpeg artifacts, watermark, text, signature, deformed, bad anatomy, extra fingers",
    },
    # Fast drafts (1-4 steps, no CFG). Non-commercial licence.
    "turbo": {
        "model": "stabilityai/sdxl-turbo",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance": 0.0,
        "negative": "",
    },
}

SDXL_FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "image"


def load_pipeline(model: str, offload: bool):
    import torch
    from diffusers import AutoencoderKL, AutoPipelineForText2Image

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available - check the PyTorch install in venv.")

    vae = AutoencoderKL.from_pretrained(SDXL_FP16_VAE, dtype=torch.float16)
    pipe = AutoPipelineForText2Image.from_pretrained(
        model, vae=vae, dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    if offload:
        # Moves each sub-model to the GPU only while it runs; fits SDXL into 8 GB.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def save_image(image, meta: dict) -> Path:
    from PIL.PngImagePlugin import PngInfo

    now = datetime.now()
    day_dir = OUTPUTS / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{now.strftime('%H%M%S')}_{meta['seed']}_{slugify(meta['prompt'])}.png"

    info = PngInfo()
    info.add_text("parameters", json.dumps(meta, ensure_ascii=False))
    image.save(path, pnginfo=info)
    return path


def append_gallery(entries: list[tuple[Path, dict]]) -> None:
    lines = []
    for path, meta in reversed(entries):
        rel = path.relative_to(ROOT).as_posix()
        lines.append(
            f"### {meta['created']} · seed {meta['seed']} · {meta['preset']}\n\n"
            f"**Prompt:** {meta['prompt']}\n\n"
            f'<img src="{rel}" width="512">\n'
        )
    header = "" if GALLERY.exists() else "# Gallery\n\nNewest first.\n\n"
    old = GALLERY.read_text(encoding="utf-8") if GALLERY.exists() else ""
    if old.startswith("# Gallery"):
        head, _, rest = old.partition("\n\nNewest first.\n\n")
        GALLERY.write_text(head + "\n\nNewest first.\n\n" + "\n".join(lines) + "\n" + rest, encoding="utf-8")
    else:
        GALLERY.write_text(header + "\n".join(lines) + "\n" + old, encoding="utf-8")


def git_push(paths: list[Path], message: str) -> None:
    files = [str(p.relative_to(ROOT)) for p in paths] + [GALLERY.name]
    subprocess.run(["git", "add", "--", *files], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    subprocess.run(["git", "push"], cwd=ROOT, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Local SDXL image generator")
    p.add_argument("prompt", nargs="?", help="text prompt")
    p.add_argument("--prompts-file", type=Path, help="file with one prompt per line (# = comment)")
    p.add_argument("--preset", choices=PRESETS, default="sdxl")
    p.add_argument("--model", help="override Hugging Face model id (any SDXL checkpoint)")
    p.add_argument("--negative", help="negative prompt (overrides preset default)")
    p.add_argument("-W", "--width", type=int)
    p.add_argument("-H", "--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int, help="base seed (random if omitted)")
    p.add_argument("-n", "--count", type=int, default=1, help="images per prompt")
    p.add_argument("--no-offload", action="store_true", help="keep whole model on GPU (needs >10 GB VRAM)")
    p.add_argument("--push", action="store_true", help="git commit + push the new images")
    args = p.parse_args()
    if not args.prompt and not args.prompts_file:
        p.error("give a prompt or --prompts-file")
    return args


def main():
    args = parse_args()
    import torch

    preset = PRESETS[args.preset]
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        for line in args.prompts_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                prompts.append(line.strip())

    settings = {
        "preset": args.preset,
        "model": args.model or preset["model"],
        "negative": preset["negative"] if args.negative is None else args.negative,
        "width": args.width or preset["width"],
        "height": args.height or preset["height"],
        "steps": args.steps or preset["steps"],
        "guidance": preset["guidance"] if args.guidance is None else args.guidance,
    }

    print(f"Loading {settings['model']} ...", flush=True)
    pipe = load_pipeline(settings["model"], offload=not args.no_offload)

    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    saved = []
    for pi, prompt in enumerate(prompts):
        for i in range(args.count):
            seed = (base_seed + pi * args.count + i) % 2**32
            start = time.time()
            image = pipe(
                prompt=prompt,
                negative_prompt=settings["negative"] or None,
                width=settings["width"],
                height=settings["height"],
                num_inference_steps=settings["steps"],
                guidance_scale=settings["guidance"],
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            meta = {
                "prompt": prompt,
                "seed": seed,
                **settings,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            path = save_image(image, meta)
            saved.append((path, meta))
            print(f"[{time.time() - start:.1f}s] {path.relative_to(ROOT)}", flush=True)

    append_gallery(saved)

    if args.push:
        first = prompts[0][:60]
        git_push([p for p, _ in saved], f"Generate {len(saved)} image(s): {first}")


if __name__ == "__main__":
    main()
