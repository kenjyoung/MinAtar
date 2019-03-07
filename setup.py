from distutils.core import setup

setup(name='MinAtar',
      version='1.0.1',
      description='A miniaturized version of the arcade learning environment.',
      url='https://github.com/kenjyoung/MinAtar',
      author='Kenny Young',
      author_email='kjyoung@ualberta.com',
      license='GPL',
      packages=['minatar', 'minatar.environments'],
      package_dir={'minatar': 'minatar_environment', 'minatar.environments':'environments'})