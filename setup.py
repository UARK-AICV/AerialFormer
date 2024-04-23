# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


if __name__ == '__main__':
    setup(
        name='aerialseg',
        version='0.0.1',
        # description='Open MMLab Semantic Segmentation Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='AICV Lab, University of Arkansas',
        keywords='computer vision, semantic segmentation',
        # url='http://github.com/open-mmlab/mmsegmentation',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        license='Apache License 2.0',
        install_requires=[
        'yapf==0.40.1'
        ],
        # install_requires=parse_requirements('requirements/runtime.txt'),
        # extras_require={
        #     'all': parse_requirements('requirements.txt'),
        #     'tests': parse_requirements('requirements/tests.txt'),
        #     'build': parse_requirements('requirements/build.txt'),
        #     'optional': parse_requirements('requirements/optional.txt'),
        #     'mim': parse_requirements('requirements/mminstall.txt'),
        # },
        ext_modules=[],
        zip_safe=False)
