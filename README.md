# SWV Tool

Self-contained Flask web app for downward-peak SWV baseline correction.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000`.

## Render deployment

This repo includes [render.yaml](/Users/chenshudevs/Documents/SWV_tool/render.yaml), so Render can detect the service settings automatically.

### Deploy steps

1. Push this project to GitHub.
2. In Render, choose `New +` -> `Blueprint`.
3. Connect the GitHub repo.
4. Render will read `render.yaml` and create the web service.
5. After the first deploy finishes, open the generated `.onrender.com` URL.

### Runtime details

- Web server: `gunicorn`
- Bind address: `0.0.0.0:$PORT`
- Health check: `/healthz`
- Matplotlib cache: `/tmp/mplconfig`

### Important note

The current app stores uploaded files and generated CSV outputs in memory (`UPLOAD_CACHE`, `OUTPUT_CACHE`). That is acceptable for a simple single-service deployment, but it is not durable storage. If the service restarts, cached uploads and downloads are lost.
