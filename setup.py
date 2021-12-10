from setuptools import setup, find_packages


install_deps = ['napari',
                'napari-plugin-engine>=0.1.4',
                'vollseg',
                'imagecodecs']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vollseg-napari',
    author='Varun Kapoor',
    author_email='varun.kapoor@kapoorlabs.org',
    license='BSD-3',
    url='https://github.com/kapoorlab/vollseg-napari',
    description='Segmentation tool for irregular cell shapes in 2/3D',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=install_deps
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Framework :: napari',
    ],
    entry_points={
        'napari.plugin': [
            'vollseg-napari = vollseg_napari',
        ],
    },
)

