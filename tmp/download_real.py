import os, urllib.request, sys

out_dir = r"data\benchmark\real"
os.makedirs(out_dir, exist_ok=True)

images = {
    "real_0_portrait.jpg": "https://picsum.photos/id/64/800/800",
    "real_1_landscape.jpg": "https://picsum.photos/id/1015/800/600",
    "real_2_product.jpg": "https://picsum.photos/id/180/800/800",
    "real_3_indoor.jpg": "https://picsum.photos/id/1080/800/600",
    "real_4_outdoor.jpg": "https://picsum.photos/id/1043/800/600",
}

ok = 0
for name, url in images.items():
    dest = os.path.join(out_dir, name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        print(f"OK {name} {len(data)} bytes")
        ok += 1
    except Exception as e:
        print(f"FAIL {name}: {e}")

print(f"Downloaded {ok}/{len(images)}")
