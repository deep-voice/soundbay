import setuptools
from setuptools import setup
from distutils.util import convert_path
import os

main_ns = {}
ver_path = convert_path('soundbay/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)
VER = main_ns['__version__']

requirementPath = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='soundbay',
    version=VER,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    license_files='LICENCE',
    description='Deep Learning Framework for Bioacoustics',
    author='soundbay',
    author_email='info@deepvoicefoundation.com',
    url='https://github.com/deep-voice/soundbay',
    keywords=['Bioacoustics', 'Machine Learning'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)
