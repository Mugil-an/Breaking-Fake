# Frontend (Breaking-Fake)

Run the development frontend (Vite + React):

1. Install dependencies

```bash
cd frontend
npm install
```

2. Copy `.env.example` to `.env` and set `VITE_API_KEY` to match backend

3. Start dev server

```bash
npm run dev
```

The app proxies `/api` to `http://localhost:8000` by default; adjust `VITE_API_BASE` if needed.
Frontend placeholder.

If you add a SPA (React/Vue/Angular), place code under `frontend/` and provide a `Dockerfile` and `package.json` there.
