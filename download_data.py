import urllib.request
import zipfile
import os

print("Creating data folders...")
os.makedirs("data/raw/setA", exist_ok=True)
os.makedirs("data/raw/setB", exist_ok=True)

print("Downloading Set A - 20,000 patients (takes 2-3 mins)...")
urllib.request.urlretrieve(
    "https://physionet.org/static/published-projects/challenge-2019/training_setA.zip",
    "data/raw/training_setA.zip"
)
print("Set A downloaded.")

print("Downloading Set B - 20,000 patients (takes 2-3 mins)...")
urllib.request.urlretrieve(
    "https://physionet.org/static/published-projects/challenge-2019/training_setB.zip",
    "data/raw/training_setB.zip"
)
print("Set B downloaded.")

print("Extracting Set A...")
with zipfile.ZipFile("data/raw/training_setA.zip", "r") as z:
    z.extractall("data/raw/setA")

print("Extracting Set B...")
with zipfile.ZipFile("data/raw/training_setB.zip", "r") as z:
    z.extractall("data/raw/setB")

print("Removing zip files...")
os.remove("data/raw/training_setA.zip")
os.remove("data/raw/training_setB.zip")

print("Counting patients...")
setA = len([f for f in os.listdir("data/raw/setA") if f.endswith(".psv")])
setB = len([f for f in os.listdir("data/raw/setB") if f.endswith(".psv")])
print(f"Set A: {setA} patient files")
print(f"Set B: {setB} patient files")
print(f"Total: {setA + setB} patients")
print("All done!")