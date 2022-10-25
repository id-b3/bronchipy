from setuptools import setup, find_packages

VERSION = '0.0.3'
DESCRIPTION = 'Supporting tools for processing the output of the AirFlow'\
    ' pipeline.'
LONG_DESCRIPTION = 'Scripts and functions to process the output of AirFlow'\
    ' pipeline, for use with the output files. Allows for post-processing'\
    ' of the data tables.'

# Setting up
setup(name="bronchipy",
      version=VERSION,
      author="ImaLife",
      author_email="<i.dudurych@umcg.nl>",
      description=DESCRIPTION,
      long_description_content_type="text/markdown",
      long_description=LONG_DESCRIPTION,
      packages=find_packages(),
      install_requires=['nibabel==3.1.0', 'pydicom==1.4.2', 'numpy==1.19.2',
                        'pandas==1.4.3', 'scipy==1.5.3',
                        'scikit-learn==0.23.1', 'matplotlib==3.3.4',
                        'SimpleITK==1.2.4', 'scikit-image==0.17.2'],
      keywords=[
          'python', 'analysis', 'medical', 'imaging', 'research'
      ],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Operating System :: Unix",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
      ])
