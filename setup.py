from setuptools import setup

setup(
    name='MinAtar',
    version='1.0.6',
    description='A miniaturized version of the arcade learning environment.',
    url='https://github.com/kenjyoung/MinAtar',
    author='Kenny Young',
    author_email='kjyoung@ualberta.com',
    license='GPL',
    packages=['minatar', 'minatar.environments'],
    install_requires=[
        'cycler>=0.10.0',
        'kiwisolver>=1.0.1',
        'matplotlib>=3.0.3',
        'numpy>=1.16.2',
        'pandas>=0.24.2',
        'pyparsing>=2.3.1',
        'python-dateutil>=2.8.0',
        'pytz>=2018.9',
        'scipy>=1.2.1',
        'seaborn>=0.9.0',
        'six>=1.12.0',
        'torch>=1.0.0',
    ])
