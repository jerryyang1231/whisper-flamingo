import tempfile
with tempfile.NamedTemporaryFile(dir="/tmp") as f:
    print(f.name)
