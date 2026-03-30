# 📄 DocScan Pro — Document Cleaner

## Folder Structure
```
project/
├── app.py
├── requirements.txt
├── Procfile
└── templates/
    └── index.html
```

## Local Chalane Ke Liye
```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

## PDF Support (Optional)
Windows mein PDF support ke liye:
1. Download: https://github.com/oschwartz10612/poppler-windows/releases
2. Unzip aur `bin/` folder ko PATH mein add karein

Linux/Mac:
```bash
sudo apt install poppler-utils   # Ubuntu
brew install poppler             # Mac
```

## Live Deploy — Railway (Free)
1. GitHub pe push karein
2. https://railway.app → New Project → Deploy from GitHub
3. Done! Auto-deploy hoga

## Live Deploy — Render (Free)
1. https://render.com → New Web Service
2. GitHub repo connect karein
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

## Live Deploy — Heroku
```bash
heroku create docscan-pro
git push heroku main
heroku open
```

## Features
- ✅ Koi file save nahi (fully in-memory)
- ✅ 4 cleaning modes: Standard, Scan, Soft, Grayscale
- ✅ Auto deskew (tedi image seedhi karo)
- ✅ Auto border crop
- ✅ Denoise + Sharpen
- ✅ Camera scan (mobile/laptop)
- ✅ Before/After preview + Full modal
- ✅ Print single / Print all
- ✅ ZIP download
- ✅ PDF support
- ✅ Refresh ke baad sab clear
