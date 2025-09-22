from setuptools import setup, find_packages

setup(
    name='dex_retargeting',
    version='0.1.0',
    description='A library for dexterous hand retargeting and optimization',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/dex-retargeting',  # optional
    license='MIT',  # or the actual license
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    # install_requires=[
    #     'numpy',
    #     'torch',
    #     'pyyaml',
    #     # add other dependencies here if needed
    # ],
    python_requires='>=3.7',
)