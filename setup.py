from setuptools import setup

setup(name='acousticIndices',
      version='0.5',
      description='Audio indexing library, with common biological indices.',
      url='https://github.com/dkadish/Acoustic_Indices',
      author='Patrice Guyot, Alice Eldridge, Mika Peck, David Kadish',
      author_email='guyot.patrice@gmail.com',
      license='GPLv3',
      packages=['acousticIndices'],
      zip_safe=True,
      install_requires=[
            'numpy>=1.5.0',
            'scipy>=0.9.0',
            'cached-property>=1.0.0',
            'PyAudio>=0.2.6',
        ],
      )