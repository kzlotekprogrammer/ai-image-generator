# AI Image Generator

Local SDXL image generation on the user's RTX 3070 (8 GB). The user drives this repo via Claude Code Remote Control (often from a phone) and talks in Polish — reply in Polish.

## Generating on request

Run from the repo root (PowerShell):

```powershell
.\venv\Scripts\python.exe generate.py "<prompt in English>" --push
```

- Translate/expand the user's request into a good English SDXL prompt (subject, style, lighting, composition).
- `--preset turbo -n 4` for quick drafts/variations; default `sdxl` for final quality.
- `--push` commits the PNGs + `GALLERY.md` and pushes, so the user can view them on GitHub. Always push unless told otherwise.
- After pushing, also send the image(s) to the user's device with SendUserFile when available.
- Reuse `--seed` from PNG metadata / GALLERY.md to iterate on an image the user liked.

## Notes

- Models are cached in `models/` (gitignored). First run of a new model downloads ~7 GB.
- Norton intercepts HTTPS; `truststore` in generate.py makes Python use the Windows cert store. Don't disable SSL verification.
- VRAM is tight: model CPU offload is on by default. Close nothing on the user's machine without asking.
