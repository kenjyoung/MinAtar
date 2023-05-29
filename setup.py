from setuptools import setup

packages = ['minatar', 'minatar.environments']
install_requires = [
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
]

examples_requires = [
    'torch>=1.0.0',
]

entry_points = {
    'gymnasium.envs': ['MinAtar=minatar.gym:register_envs']
}

setup(
    name='MinAtar',
    version='1.0.15',
    description='A miniaturized version of the Arcade Learning Environment.',
    url='https://github.com/kenjyoung/MinAtar',
    author='Kenny Young',
    author_email='kjyoung@ualberta.ca',
    license='GPL',
    packages=packages,
    entry_points=entry_points,
    install_requires=install_requires,
    extras_require={'examples': examples_requires},
)
