from setuptools import setup, find_packages
from os import path



_dir = path.dirname(__file__)

with open(path.join(_dir,'vollseg_napari','_version.py'), encoding="utf-8") as f:
    exec(f.read())

with open(path.join(_dir,'README.md'), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='vollseg-napari',

    version= '2.0.6',

    description='Irregular cell shape segmentation using VollSeg',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kapoorlab/vollseg',
    project_urls={
        'Source Code': 'https://github.com/kapoorlab/vollseg-napari',
        'Documentation': 'https://github.com/kapoorlab/vollseg-napari',
        'Bug Tracker': 'https://github.com/kapoorlab/vollseg-napari/issues',
        'User Support': 'https://forum.image.sc/tag/vollseg',
        'Twitter': 'https://twitter.com/entracod',
    },
    author='Varun Kapoor',
    author_email='varun.kapoor@kapoorlabs.org',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.7',

      package_data={'vollseg_napari': [ 'resources/*' ]},

      entry_points={'napari.plugin': 'VollSeg = vollseg_napari'},
      
      dependeny_links = ['https://github.com/bhoeckendorf/pyklb.git@skbuild'],

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
        'vollseg',
        'tensorflow-gpu==2.7.0',
        'napari>=0.4.9',
         'typing-extensions>=3.10.0.0'
        'magicgui>=0.3.0'
    ],
)
