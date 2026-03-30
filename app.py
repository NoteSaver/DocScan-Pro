from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import zipfile
import io
import base64
import time
import threading
import os
import sys

# ─── OCR check ────────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image as PILImage

    # Common Windows Tesseract paths — auto-detect
    if sys.platform == "win32":
        _win_tess_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.environ.get("LOCALAPPDATA",""),
                         r"Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for _p in _win_tess_paths:
            if os.path.isfile(_p):
                pytesseract.pytesseract.tesseract_cmd = _p
                break

    OCR_AVAILABLE = True
    # Get available languages once at startup
    try:
        OCR_LANGS = pytesseract.get_languages()
    except Exception:
        OCR_LANGS = ["eng"]
except ImportError:
    OCR_AVAILABLE = False
    OCR_LANGS     = []

app = Flask(__name__)

# ─── In-Memory Store ──────────────────────────────────────────────────────────
store      = {}
store_lock = threading.Lock()

def cleanup_old():
    now = time.time()
    with store_lock:
        expired = [k for k,v in store.items() if now - v.get("created_at", now) > 3600]
        for k in expired:
            del store[k]


# ─── PDF → Images  (multiple fallbacks, no Poppler needed) ───────────────────
def pdf_to_images(pdf_bytes):
    """
    Try 3 methods in order:
      1. PyMuPDF  (fitz)   — best, no external binary needed
      2. pdf2image          — needs poppler; auto-detects Windows path
      3. Pillow PDF         — basic fallback
    Returns list of (name, jpeg_bytes) or None on total failure.
    """

    # ── Method 1: PyMuPDF (fitz) ──────────────────────────────────────────────
    try:
        import fitz  # PyMuPDF
        doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
        result = []
        for i, page in enumerate(doc):
            mat  = fitz.Matrix(2.5, 2.5)          # 250 dpi
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            result.append((f"page_{i+1}.jpg", img_bytes))
        doc.close()
        return result
    except ImportError:
        pass
    except Exception as e:
        print(f"[PyMuPDF error] {e}", file=sys.stderr)

    # ── Method 2: pdf2image + poppler ─────────────────────────────────────────
    try:
        from pdf2image import convert_from_bytes
        from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

        # Common Windows poppler paths — add yours here if needed
        win_paths = [
            r"C:\poppler\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin",
            os.path.join(os.environ.get("USERPROFILE",""), r"Downloads\poppler\bin"),
            os.path.join(os.environ.get("USERPROFILE",""), r"Desktop\poppler\bin"),
        ]

        poppler_path = None
        if sys.platform == "win32":
            for p in win_paths:
                if os.path.isdir(p):
                    poppler_path = p
                    break

        kwargs = dict(dpi=250)
        if poppler_path:
            kwargs["poppler_path"] = poppler_path

        pages  = convert_from_bytes(pdf_bytes, **kwargs)
        result = []
        for i, page in enumerate(pages):
            buf = io.BytesIO()
            page.save(buf, "JPEG", quality=95)
            result.append((f"page_{i+1}.jpg", buf.getvalue()))
        return result

    except ImportError:
        pass
    except Exception as e:
        print(f"[pdf2image error] {e}", file=sys.stderr)

    # ── Method 3: Pillow direct ───────────────────────────────────────────────
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(pdf_bytes))
        result = []
        for i in range(getattr(img, "n_frames", 1)):
            try:
                img.seek(i)
            except EOFError:
                break
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=95)
            result.append((f"page_{i+1}.jpg", buf.getvalue()))
        if result:
            return result
    except Exception as e:
        print(f"[Pillow error] {e}", file=sys.stderr)

    return None   # All methods failed


def pdf_error_message():
    """Return a helpful install message based on platform."""
    if sys.platform == "win32":
        return (
            "Install PyMuPDF for PDF support: <br>"
            "<code>pip install pymupdf</code><br><br>"
            "Or download Poppler: "
            "<a href='https://github.com/oschwartz10612/poppler-windows/releases' target='_blank'>"
            "poppler-windows releases</a> → unzip → "
            "<code>bin/</code> folder ka path <code>app.py</code> ke "
            "<code>win_paths</code> list mein add karein."
        )
    return "PDF support: <code>pip install pymupdf</code>"


# ─── Image Processing ─────────────────────────────────────────────────────────
def process_image(img_bytes, mode="standard", brightness=0, contrast=1.0,
                  sharpness=False, denoise=False, deskew=False, border_remove=False):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Deskew
    if deskew:
        gray_d   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_d   = cv2.bitwise_not(gray_d)
        thresh_d = cv2.threshold(gray_d, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords   = np.column_stack(np.where(thresh_d > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if abs(angle) > 0.5:
                h, w = img.shape[:2]
                M   = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

    # Border remove
    if border_remove:
        gray_b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw  = cv2.threshold(gray_b, 200, 255, cv2.THRESH_BINARY_INV)
        conts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if conts:
            x, y, w, h = cv2.boundingRect(max(conts, key=cv2.contourArea))
            pad = 10
            img = img[max(0,y-pad):min(img.shape[0],y+h+pad),
                      max(0,x-pad):min(img.shape[1],x+w+pad)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg      = cv2.medianBlur(dilated, 21)
    diff    = 255 - cv2.absdiff(gray, bg)
    norm    = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    if brightness != 0 or contrast != 1.0:
        norm = cv2.convertScaleAbs(norm, alpha=contrast, beta=brightness)
    if denoise:
        norm = cv2.fastNlMeansDenoising(norm, h=10)

    if mode == "scan":
        thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 21, 10)
    elif mode == "grayscale":
        thresh = norm
    elif mode == "soft":
        thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 51, 20)
    else:
        thresh = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 15)

    if sharpness:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        thresh = cv2.filter2D(thresh, -1, kernel)

    if mode != "grayscale":
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    _, buf = cv2.imencode(".png", thresh)
    return buf.tobytes()


def to_thumb(img_bytes, max_w=400):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def get_info(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {}
    h, w = img.shape[:2]
    return {"width": w, "height": h, "size_kb": round(len(img_bytes)/1024, 1)}


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    cleanup_old()
    files         = request.files.getlist("images")
    brightness    = int(request.form.get("brightness", 0))
    contrast      = float(request.form.get("contrast", 1.0))
    mode          = request.form.get("mode", "standard")
    sharpness     = request.form.get("sharpness") == "true"
    denoise       = request.form.get("denoise") == "true"
    deskew        = request.form.get("deskew") == "true"
    border_remove = request.form.get("border_remove") == "true"

    results = []
    for file in files:
        if not file.filename:
            continue
        raw = file.read()
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

        if ext == "pdf":
            pages = pdf_to_images(raw)
            if pages is None:
                results.append({"error": pdf_error_message()})
                continue
            items = pages
        else:
            items = [(file.filename, raw)]

        for (name, img_bytes) in items:
            cleaned = process_image(img_bytes, mode, brightness, contrast,
                                    sharpness, denoise, deskew, border_remove)
            if cleaned is None:
                results.append({"error": f"Could not read image '{name}'."})
                continue

            token    = f"{int(time.time()*1000)}_{name}"
            out_name = "clean_" + name.rsplit(".", 1)[0] + ".png"

            with store_lock:
                store[token] = {"filename": out_name, "data": cleaned,
                                "created_at": time.time()}

            results.append({
                "token":         token,
                "original_name": name,
                "download_name": out_name,
                "before":        to_thumb(img_bytes),
                "after":         to_thumb(cleaned),
                "after_full":    base64.b64encode(cleaned).decode(),
                "info_in":       get_info(img_bytes),
                "info_out":      get_info(cleaned),
            })

    return jsonify(results)


@app.route("/download/<path:token>")
def download_single(token):
    entry = store.get(token)
    if not entry:
        return "File not found or session expired.", 404
    return send_file(io.BytesIO(entry["data"]), mimetype="image/png",
                     as_attachment=True, download_name=entry["filename"])


@app.route("/download-zip", methods=["POST"])
def download_zip():
    tokens = request.json.get("tokens", [])
    buf    = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in tokens:
            e = store.get(t)
            if e:
                zf.writestr(e["filename"], e["data"])
    buf.seek(0)
    return send_file(buf, mimetype="application/zip",
                     as_attachment=True, download_name="cleaned_documents.zip")


# ─── Text Extraction (OCR) ────────────────────────────────────────────────────
def extract_text_from_bytes(img_bytes, lang="eng", psm=6):
    """Run Tesseract OCR on cleaned image bytes. Returns extracted text."""
    if not OCR_AVAILABLE:
        return None, "pytesseract not installed"

    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, "Image decode failed"

        # Extra pre-processing for better OCR accuracy
        # Ensure white background, black text
        if np.mean(img) < 128:
            img = cv2.bitwise_not(img)

        # Upscale small images for better accuracy
        h, w = img.shape[:2]
        if w < 1000:
            scale = 1000 / w
            img = cv2.resize(img, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_CUBIC)

        pil_img = PILImage.fromarray(img)
        config  = f"--psm {psm} --oem 3"
        text    = pytesseract.image_to_string(pil_img, lang=lang, config=config)
        return text.strip(), None

    except pytesseract.TesseractNotFoundError:
        return None, (
            "Tesseract binary not found.<br>"
            "Windows: <a href='https://github.com/UB-Mannheim/tesseract/wiki' target='_blank'>"
            "Download Tesseract</a> → install → restart app"
        )
    except Exception as e:
        return None, str(e)


@app.route("/ocr-info")
def ocr_info():
    """Return OCR availability and languages."""
    return jsonify({
        "available": OCR_AVAILABLE,
        "languages":  OCR_LANGS,
    })


@app.route("/extract-text", methods=["POST"])
def extract_text():
    """
    Accepts:
      - token  (already-processed cleaned image in store)
      - lang   (tesseract language code, default 'eng')
      - psm    (page segmentation mode, default 6)
    Returns extracted text.
    """
    data  = request.json or {}
    token = data.get("token", "")
    lang  = data.get("lang", "eng")
    psm   = int(data.get("psm", 6))

    if not OCR_AVAILABLE:
        return jsonify({"error": "pytesseract not installed — run: pip install pytesseract"}), 400

    entry = store.get(token)
    if not entry:
        return jsonify({"error": "Image not found. Please clean the image first."}), 404

    text, err = extract_text_from_bytes(entry["data"], lang=lang, psm=psm)
    if err:
        return jsonify({"error": err}), 500

    return jsonify({
        "text":     text,
        "chars":    len(text),
        "words":    len(text.split()) if text else 0,
        "lines":    len([l for l in text.splitlines() if l.strip()]) if text else 0,
        "filename": entry["filename"],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "stored": len(store), "platform": sys.platform})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)