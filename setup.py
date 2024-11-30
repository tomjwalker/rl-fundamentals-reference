import os
import sys
from pathlib import Path

from setuptools import Extension, setup, find_packages
package_dir = os.path.join(os.path.dirname(__file__))
setup(
    name="rl",
    version="0.0.1",
    packages=find_packages(
        include=["rl", "rl.*"]
    ),
    package_dir={"": package_dir},
    zip_safe=False,
    install_requires=["numpy","moviepy", "matplotlib", "pandas", "scipy", "seaborn","gymnasium","pipreqs"],
    python_requires=">=3.7",
)
