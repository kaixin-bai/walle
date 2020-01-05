from setuptools import setup

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name='walle',
  version='0.1.1',
  author='Kevin Zakka',
  author_email="kevinarmandzakka@gmail.com",
  description='Deep Robotics Research',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url='https://github.com/kevinzakka/walle',
  python_requires='>=3.6.0',
  keywords='ai robotics',
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  install_requires=[
    'numpy>=1.0.0,<2.0.0',
    'scikit-image>=0.13.0,<2.0.0',
  ],
  extras_require={
    "rs2": ["pyrealsense2>=2.22"],
  }
)