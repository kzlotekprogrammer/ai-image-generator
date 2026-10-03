# AI Image Generator

Local SDXL image generation on the user's RTX 3070 (8 GB). The user drives this repo via Claude Code Remote Control (often from a phone) and talks in Polish — reply in Polish.

## Generating on request

Run from the repo root (PowerShell):

```powershell
.\venv\Scripts\python.exe generate.py "<prompt in English>" --push
```

- Translate/expand the user's request into a good English SDXL prompt (subject, style, lighting, composition).
- `--preset turbo -n 4` for quick drafts/variations; default `sdxl` for final quality; `--preset cartoon` (DreamShaper XL) for cartoon/mascot/sticker styles — much better than base SDXL there.
- CLIP reads only ~77 tokens: keep prompts short, put the important parts first (anything past the limit is silently dropped — check the log for "truncated").
- `--push` commits the PNGs + `GALLERY.md` and pushes, so the user can view them on GitHub. Always push unless told otherwise.
- After pushing, also send the image(s) to the user's device with SendUserFile when available.
- Reuse `--seed` from PNG metadata / GALLERY.md to iterate on an image the user liked.
- To fix one part of an image (hands, legs, an object) use `inpaint.py <png> "<what goes there>" --box x1,y1,x2,y2 --preset <same as source> -n 4 [--push]` — repaints only the box, the rest stays pixel-identical.

## Notes

- Models are cached in `models/` (gitignored). First run of a new model downloads ~7 GB.
- Norton intercepts HTTPS; `truststore` in generate.py makes Python use the Windows cert store, and this repo has `git config http.sslBackend schannel` for the same reason. Don't disable SSL verification.
- Remote: https://github.com/kzlotekprogrammer/ai-image-generator (branch `main`).
- VRAM is tight: model CPU offload is on by default. Close nothing on the user's machine without asking.
