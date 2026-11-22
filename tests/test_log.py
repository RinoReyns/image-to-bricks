"""run_log.txt: console output (stdout + stderr) is copied to <outdir>/run_log.txt, the command is the first line."""

import io
import os
import sys
import tempfile

import support

support.stub_mesh_libraries()
from image_to_bricks import cli as mtb

d = os.path.join(tempfile.mkdtemp(), "new_out")  # folder does not exist yet
real_out, real_err, real_argv = sys.stdout, sys.stderr, sys.argv
cap = io.StringIO()
sys.stdout = cap
sys.stderr = io.StringIO()
sys.argv = ["model_to_bricks.py", "in file.glb", "--x"]
log = mtb.start_log(d)
print("hello")
print("warn", file=sys.stderr)
sys.stdout, sys.stderr, sys.argv = real_out, real_err, real_argv
log.close()
txt = open(os.path.join(d, "run_log.txt"), encoding="utf-8").read()
ok = [
    txt.startswith('python model_to_bricks.py "in file.glb" --x'),
    "hello" in txt,
    "warn" in txt,
    "hello" in cap.getvalue(),
]
print(ok)
print("all passed" if all(ok) else "FAILED")
sys.exit(0 if all(ok) else 1)
