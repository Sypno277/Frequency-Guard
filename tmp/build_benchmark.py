import os, urllib.request, shutil, csv, io
from PIL import Image
import numpy as np

ROOT = r"data\benchmark"
REAL = os.path.join(ROOT, "real")
AI = os.path.join(ROOT, "ai")
out_dirs = {
    "jpeg60": os.path.join(ROOT, "jpeg60"),
    "resize": os.path.join(ROOT, "resize"),
    "screenshot": os.path.join(ROOT, "screenshot"),
}
os.makedirs(REAL, exist_ok=True)
os.makedirs(AI, exist_ok=True)
for d in out_dirs.values():
    os.makedirs(d, exist_ok=True)

def fetch(url, dest, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            print(f"OK {os.path.basename(dest)} {len(data)}b")
            return True
        except Exception as e:
            print(f"  retry{i+1} {os.path.basename(dest)}: {e}")
    return False

# Retry failed real photos
retry = {
    "real_1_landscape.jpg": "https://picsum.photos/id/1015/800/600",
    "real_3_indoor.jpg": "https://picsum.photos/id/1080/800/600",
}
for name, url in retry.items():
    if not os.path.exists(os.path.join(REAL, name)):
        fetch(url, os.path.join(REAL, name))

# AI images: use the one genuinely generated image + copy demo synthetic fakes as additional
# AI examples (labeled as procedural, not representative).
ai_srcs = [
    (os.path.join(AI, "ai_0_portrait.png"), "generated-ai"),
]
# Copy 2 demo fake images as additional AI-class samples (procedural proxy)
demo_fakes = [
    r"data\demo_dataset\fake\fake_0000.png",
    r"data\demo_dataset\fake\fake_0001.png",
    r"data\demo_dataset\fake\fake_0002.png",
]
for i, src in enumerate(demo_fakes, start=1):
    dst = os.path.join(AI, f"ai_{i}_proxy.png")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        ai_srcs.append((dst, "procedural-proxy"))

# Build degradation variants for every base image (real + ai)
def make_variants(src_path, label_source, manifest_rows, image_id):
    img = Image.open(src_path).convert("RGB")
    base = os.path.splitext(os.path.basename(src_path))[0]
    stem = f"{image_id}"
    # 1) jpeg60
    jpath = os.path.join(out_dirs["jpeg60"], f"{stem}_jpeg60.jpg")
    img.save(jpath, "JPEG", quality=50)
    manifest_rows.append((stem + "_jpeg60.jpg", "real" if label_source.startswith("real") else "ai", "jpeg60", image_id, "JPEG q=50", "data/benchmark/jpeg60/" + os.path.basename(jpath)))
    # 2) resize down then up
    rpath = os.path.join(out_dirs["resize"], f"{stem}_resize.png")
    small = img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS)
    up = small.resize((img.width, img.height), Image.LANCZOS)
    up.save(rpath, "PNG")
    manifest_rows.append((stem + "_resize.png", "real" if label_source.startswith("real") else "ai", "resize", image_id, "50% down then up", "data/benchmark/resize/" + os.path.basename(rpath)))
    # 3) screenshot simulation: add UI chrome bars + jpeg-like recompress
    spath = os.path.join(out_dirs["screenshot"], f"{stem}_screenshot.png")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + 80), (240, 240, 240))
    canvas.paste(img, (0, 0))
    # add a fake top bar and bottom bar with text blocks
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, w, 40], fill=(30, 30, 30))
    d.rectangle([0, h, w, h + 40], fill=(30, 30, 30))
    d.ellipse([12, 10, 30, 28], fill=(255, 0, 0))
    d.rectangle([40, 12, 200, 28], fill=(90, 90, 90))
    canvas.save(spath, "PNG")
    manifest_rows.append((stem + "_screenshot.png", "real" if label_source.startswith("real") else "ai", "screenshot", image_id, "screenshot w/ UI bars", "data/benchmark/screenshot/" + os.path.basename(spath)))

manifest_rows = []
# Real images
for fname in sorted(os.listdir(REAL)):
    if fname.endswith((".jpg", ".png")):
        p = os.path.join(REAL, fname)
        image_id = os.path.splitext(fname)[0]
        # pristine manifest
        manifest_rows.append((fname, "real", "pristine", image_id, "camera photo (picsum)", "data/benchmark/real/" + fname))
        make_variants(p, "real", manifest_rows, image_id)

# AI images
for fname in sorted(os.listdir(AI)):
    if fname.endswith((".jpg", ".png")):
        p = os.path.join(AI, fname)
        image_id = os.path.splitext(fname)[0]
        label = "ai"
        src_note = "generated-ai" if "ai_0" in fname else "procedural-proxy"
        manifest_rows.append((fname, "ai", "pristine", image_id, src_note, "data/benchmark/ai/" + fname))
        make_variants(p, "ai", manifest_rows, image_id)

# Write manifest
manifest_path = os.path.join(ROOT, "manifest.csv")
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "label", "state", "image_id", "provenance", "path"])
    for row in manifest_rows:
        w.writerow(row)

print(f"Manifest rows: {len(manifest_rows)}")
print("Real files:", sorted(os.listdir(REAL)))
print("AI files:", sorted(os.listdir(AI)))
print("Manifest written to", manifest_path)
