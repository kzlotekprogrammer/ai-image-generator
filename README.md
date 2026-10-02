# AI Image Generator

Lokalne generowanie obrazków na GPU (RTX 3070, 8 GB) przy pomocy Stable Diffusion XL i biblioteki [diffusers](https://github.com/huggingface/diffusers).
Wygenerowane obrazki trafiają do `outputs/RRRR-MM-DD/`, a podgląd z promptami do [GALLERY.md](GALLERY.md).

## Instalacja

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\pip.exe install -r requirements.txt
```

Modele pobierają się automatycznie przy pierwszym uruchomieniu do `models/` (ok. 7 GB na SDXL, ok. 7 GB na Turbo).

> Norton przechwytuje HTTPS własnym certyfikatem. Nowy pip i `generate.py` (przez `truststore`) korzystają z magazynu certyfikatów Windows, więc wszystko działa bez wyłączania weryfikacji SSL. Git w tym repo ma `http.sslBackend schannel` z tego samego powodu.

## Użycie

```powershell
# jeden obrazek, jakość (SDXL, 1024x1024, ok. 30 s)
.\venv\Scripts\python.exe generate.py "a lighthouse on a cliff at sunset, oil painting"

# 4 szybkie szkice (SDXL-Turbo, 512x512, kilka sekund)
.\venv\Scripts\python.exe generate.py "cyberpunk cat, neon" --preset turbo -n 4

# wiele promptów z pliku (jeden na linię, # = komentarz) + commit i push
.\venv\Scripts\python.exe generate.py --prompts-file prompts.txt --push
```

| Opcja | Opis |
|---|---|
| `--preset sdxl\|turbo` | `sdxl`: jakość (domyślnie), `turbo`: szybkie szkice |
| `--model ID` | dowolny checkpoint SDXL z Hugging Face, np. `SG161222/RealVisXL_V5.0` |
| `-n N` | liczba obrazków na prompt |
| `-W / -H` | szerokość / wysokość (wielokrotność 8; dla SDXL najlepiej ok. 1 MPx, np. 1216x832) |
| `--steps`, `--guidance` | liczba kroków i CFG |
| `--negative "..."` | negatywny prompt |
| `--seed N` | powtarzalny wynik |
| `--push` | `git add` + `commit` + `push` nowych obrazków i galerii |

Parametry każdego obrazka są zapisane w metadanych PNG (`parameters`).

## Licencje modeli

- SDXL base 1.0: CreativeML Open RAIL++-M (użycie komercyjne dozwolone)
- SDXL-Turbo: licencja niekomercyjna Stability AI, tylko do szkiców i eksperymentów
