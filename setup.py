from setuptools import setup, find_packages
from os import path



_dir = path.dirname(__file__)

with open(path.join(_dir,'src/vollseg_napari','_version.py'), encoding="utf-8") as f:
     exec(f.read())
     print(exec(f.read()))
with open(path.join(_dir,'README.md'), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='vollseg-napari',

    version= __version__,

    description='Irregular cell shape segmentation using VollSeg',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kapoorlab/vollseg-napari',
    project_urls={
        'Source Code': 'https://github.com/kapoorlab/vollseg-napari',
        'Documentation': 'https://github.com/kapoorlab/vollseg-napari',
        'Bug Tracker': 'https://github.com/kapoorlab/vollseg-napari/issues',
        'User Support': 'https://forum.image.sc/tag/vollseg-napari',
        'Twitter': 'https://twitter.com/entracod',
    },
    author='Varun Kapoor',
    author_email='varun.kapoor@kapoorlabs.org',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.7',

      package_data={'vollseg_napari': [ 'resources/*', 'napari.yaml' ]},

      entry_points={'napari.manifest': ['vollseg-napari = vollseg_napari:napari.yaml']},

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Operating System :: OS Independent',

          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',

          'Framework :: napari',
      ],

    install_requires=[
        "vollseg",
        'tensorflow;  platform_system!="Darwin" or platform_machine!="arm64"',
        'tensorflow-macos;  platform_system=="Darwin" and platform_machine=="arm64"',
        "napari>=0.4.13",
        "magicgui>=0.4.0",
        "pyqt6",
        "pynvml"
    ],
    extras_require={
        "test": ["pytest", "pytest-qt", "napari[pyqt]>=0.4.13"],
    },
)
