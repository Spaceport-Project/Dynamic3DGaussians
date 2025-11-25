import os, zipfile

path = "/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-09-24_3412x2500_combin2_all_test1/2025-08-06_16-09-24_3412x2500_combin2_all/params.npz"  # your renamed file

print("Path:", os.path.abspath(path))
print("Size:", os.path.getsize(path), "bytes")

with open(path, "rb") as f:
    head = f.read(4)
print("Header:", head)  # should start with b'PK\x03\x04' for a zip

# Optional: test zip integrity
try:
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
    print("Zip test:", "OK" if bad is None else f"First bad file: {bad}")
except Exception as e:
    print("zipfile error:", e)
