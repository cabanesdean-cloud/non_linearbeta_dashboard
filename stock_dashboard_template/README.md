# Nonlinear Beta Dashboard (Template)

This template ships your Preview dashboard as a real website:

- **frontend/**: static HTML/CSS/JS (deploy to Vercel/Netlify/GitHub Pages)
- **backend/**: FastAPI (deploy to Render/Fly/Railway) — fetches Stooq server-side (no CORS proxies)

## 1) Run locally

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/health

### Frontend
Open `frontend/index.html` in your browser.
Set **Backend URL** to: `http://localhost:8000` and click **Run analysis**.

> If your browser blocks local file -> localhost fetches, run a tiny static server:
```bash
cd frontend
python -m http.server 5173
```
Then open http://localhost:5173

## 2) Deploy backend on Render (quick)

- Create a new **Web Service**
- Root directory: `backend`
- Build command:
  ```bash
  pip install -r requirements.txt
  ```
- Start command:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```

After deploy, you’ll get a URL like: `https://your-service.onrender.com`

## 3) Deploy frontend on Vercel (quick)

- Create a new project (import from GitHub or drag/drop)
- Set project root to `frontend`
- It’s a static site (no build step required)

Then in the UI, set **Backend URL** to your Render URL.

## 4) Lock down CORS in production
In `backend/main.py`, replace `allow_origins=["*"]` with your frontend domain, e.g.
```py
allow_origins=["https://your-frontend.vercel.app"]
```
