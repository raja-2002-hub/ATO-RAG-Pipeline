"""Merge business pages into the main data folder."""
import csv, shutil, os

# 1. Copy business HTML files
src = "crawler/raw_html/html"
dst = "data/raw_html/html"
count = 0
for f in os.listdir(src):
    if f.endswith(".html") and not os.path.exists(os.path.join(dst, f)):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        count += 1
print(f"Copied {count} business HTML files")

# 2. Append business inventory rows
biz_rows = []
with open("crawler/raw_html/inventory.csv", "r") as f:
    for row in csv.DictReader(f):
        biz_rows.append(row)
print(f"Business inventory rows: {len(biz_rows)}")

with open("data/raw_html/inventory.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["url", "status", "title", "filename", "size"])
    for row in biz_rows:
        writer.writerow(row)

# 3. Verify
with open("data/raw_html/inventory.csv", "r") as f:
    total = sum(1 for _ in csv.DictReader(f))
print(f"Total inventory now: {total}")
