#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
from distutils.dir_util import copy_tree
from tempfile import mkstemp


def fail_if(p, msg):
    if p:
        print("error: {}".format(msg), file=sys.stderr)
        sys.exit(1)


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def touch(path):
    pathlib.Path(path).touch()


def cpdir(src, dest):
    copy_tree(src, dest, preserve_symlinks=1)


def sed(pattern, replace, source, dest=None):
    fin = open(source, "r")
    if dest:
        fout = open(dest, "w")
    else:
        fd, name = mkstemp()
        fout = open(name, "w")

    for line in fin:
        out = re.sub(pattern, replace, line)
        fout.write(out)

    fin.close()
    fout.close()
    if not dest:
        shutil.copyfile(name, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dest-dir", type=str, required=True, help="Destination directory."
    )
    parser.add_argument(
        "--binding-path", type=str, required=True, help="Path to Triton Python binding."
    )

    FLAGS = parser.parse_args()

    FLAGS.triton_version = None
    with open("TRITON_VERSION", "r") as vfile:
        FLAGS.triton_version = vfile.readline().strip()

    FLAGS.whl_dir = os.path.join(FLAGS.dest_dir, "wheel")

    print("=== Building in: {}".format(os.getcwd()))
    print("=== Using builddir: {}".format(FLAGS.whl_dir))
    print("Adding package files")
    mkdir(os.path.join(FLAGS.whl_dir, "tritonserver"))
    shutil.copy("tritonserver/__init__.py", os.path.join(FLAGS.whl_dir, "tritonserver"))
    # Type checking marker file indicating support for type checkers.
    # https://peps.python.org/pep-0561/
    shutil.copy("tritonserver/py.typed", os.path.join(FLAGS.whl_dir, "tritonserver"))
    cpdir("tritonserver/_c", os.path.join(FLAGS.whl_dir, "tritonserver", "_c"))
    cpdir("tritonserver/_api", os.path.join(FLAGS.whl_dir, "tritonserver", "_api"))
    PYBIND_LIB = os.path.basename(FLAGS.binding_path)
    shutil.copyfile(
        FLAGS.binding_path,
        os.path.join(FLAGS.whl_dir, "tritonserver", "_c", PYBIND_LIB),
    )

    shutil.copyfile("LICENSE.txt", os.path.join(FLAGS.whl_dir, "LICENSE.txt"))
    shutil.copyfile("setup.py", os.path.join(FLAGS.whl_dir, "setup.py"))
    shutil.copyfile("pyproject.toml", os.path.join(FLAGS.whl_dir, "pyproject.toml"))

    os.chdir(FLAGS.whl_dir)
    print("=== Building wheel")
    args = ["python3", "-m", "build"]

    wenv = os.environ.copy()
    wenv["VERSION"] = FLAGS.triton_version
    wenv["TRITON_PYBIND"] = PYBIND_LIB
    p = subprocess.Popen(args, env=wenv)
    p.wait()
    fail_if(p.returncode != 0, "Building wheel failed failed")

    cpdir("dist", FLAGS.dest_dir)

    print("=== Output wheel file is in: {}".format(FLAGS.dest_dir))
    touch(os.path.join(FLAGS.dest_dir, "stamp.whl"))
