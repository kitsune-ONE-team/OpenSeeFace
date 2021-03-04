#!/usr/bin/env python3
"""
Install CPU version:
$ pip install openseeface[cpu]
or
$ pip install path/to/openseeface.whl[cpu]

Install GPU version:
$ pip install openseeface[gpu]
or
$ pip install path/to/openseeface.whl[gpu]
"""

import setuptools


if __name__ == "__main__":
    setuptools.setup(
        name='openseeface',
        version='1.20.2',
        Description='OpenSeeFace',
        long_description=(
            'Robust realtime face and facial landmark tracking '
            'on CPU/GPU with Unity integration'),
        url='https://github.com/tobspr/RenderPipeline/wiki',
        download_url='https://github.com/tobspr/RenderPipeline',
        author='Emiliana',
        license='BSD-2-Clause License',
        packages=(
            'openseeface',
            'openseeface.models',
        ),
        include_package_data=True,
        install_requires=[
            'opencv-python',
            'pillow',
            'numpy',
        ],
        extras_require={
            'cpu': ['onnxruntime'],
            'gpu': ['onnxruntime-gpu'],
        },
        entry_points={
            'console_scripts': (
                'facetracker=openseeface.facetracker:main',
            ),
        },
    )
