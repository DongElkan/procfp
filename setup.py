import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

PACKAGE_DIR = "procfp"

setup(
    name="procfp",
    version="0.0.1",
    description="procfp: automatic processing liquid chromatograms "
                "for fingerprinting.",
    author="Dong Nai-ping",
    author_email="nai-ping.dong@polyu.edu.hk",
    packages=[
        "procfp",
    ],
    ext_modules=cythonize([
        os.path.join(PACKAGE_DIR, "core/*.pyx")
    ],
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[
        np.get_include()
    ]
)
