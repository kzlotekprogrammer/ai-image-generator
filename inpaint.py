"""Repaint a rectangular region of an existing image (SDXL inpainting), keep the rest untouched.

Examples:
    python inpaint.py outputs/2026-10-03/img.png "furry brown legs" --box 330,610,890,860 --preset cartoon -n 4
    python inpaint.py img.png "a red hat" --box 300,0,700,250 --strength 0.8 --push
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from generate import PRESETS, ROOT, SDXL_FP16_VAE, append_gallery, git_push, save_image


def load_pipeline(model: str, offload: bool):
    import torch
    from diffusers import AutoencoderKL, AutoPipelineForInpainting

    vae = AutoencoderKL.from_pretrained(SDXL_FP16_VAE, dtype=torch.float16)
    pipe = AutoPipelineForInpainting.from_pretrained(
        model, vae=vae, dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def make_mask(size, box, feather: int):
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask


def parse_args():
    p = argparse.ArgumentParser(description="Repaint part of an image with SDXL inpainting")
    p.add_argument("image", type=Path, help="source image")
    p.add_argument("prompt", help="what should appear in the repainted region")
    p.add_argument("--box", required=True, help="region to repaint: x1,y1,x2,y2 in pixels")
    p.add_argument("--preset", choices=PRESETS, default="sdxl")
    p.add_argument("--model", help="override Hugging Face model id")
    p.add_argument("--negative", help="negative prompt (overrides preset default)")
    p.add_argument("--strength", type=float, default=0.9, help="0-1, how much the region may change")
    p.add_argument("--feather", type=int, default=24, help="mask edge blur in pixels")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int, help="base seed (random if omitted)")
    p.add_argument("-n", "--count", type=int, default=1)
    p.add_argument("--no-offload", action="store_true")
    p.add_argument("--push", action="store_true", help="git commit + push the new images")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from PIL import Image

    preset = PRESETS[args.preset]
    source = Image.open(args.image).convert("RGB")
    box = tuple(int(v) for v in args.box.split(","))
    mask = make_mask(source.size, box, args.feather)

    settings = {
        "preset": args.preset,
        "model": args.model or preset["model"],
        "negative": preset["negative"] if args.negative is None else args.negative,
        "width": source.width,
        "height": source.height,
        "steps": args.steps or preset["steps"],
        "guidance": preset["guidance"] if args.guidance is None else args.guidance,
        "inpaint": {"source": args.image.as_posix(), "box": box, "strength": args.strength},
    }

    print(f"Loading {settings['model']} (inpainting) ...", flush=True)
    pipe = load_pipeline(settings["model"], offload=not args.no_offload)

    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    saved = []
    for i in range(args.count):
        seed = (base_seed + i) % 2**32
        start = time.time()
        result = pipe(
            prompt=args.prompt,
            negative_prompt=settings["negative"] or None,
            image=source,
            mask_image=mask,
            width=source.width,
            height=source.height,
            strength=args.strength,
            num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance"],
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        # Paste the original back outside the mask so untouched areas stay pixel-identical.
        result = Image.composite(result.resize(source.size), source, mask)
        meta = {
            "prompt": args.prompt,
            "seed": seed,
            **settings,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = save_image(result, meta)
        saved.append((path, meta))
        print(f"[{time.time() - start:.1f}s] {path.relative_to(ROOT)}", flush=True)

    append_gallery(saved)

    if args.push:
        git_push([p for p, _ in saved], f"Inpaint {len(saved)} image(s): {args.prompt[:60]}")


if __name__ == "__main__":
    main()
