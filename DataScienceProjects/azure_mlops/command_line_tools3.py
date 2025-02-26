import sys

print(f"Script name: {sys.argv[0]}")
print(f"First argument: {sys.argv[1]}")
print(f"Second argument: {sys.argv[2]}")

from setuptools import setup

# This is a minimal setup.py file
setup(
    name='mypackage',
    version='1.0',
    install_requires=['requests', 'click']
)