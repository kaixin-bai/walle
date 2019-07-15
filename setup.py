from setuptools import setup


setup(
    name='walle',
    version='0.1',
    description='Deep Robotics Research',
    author='Kevin Zakka',
    python_requires=">=3.6.0",
    keywords='ai robotics',
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
        'scikit-image>=0.13.0,<2.0.0',
        "pyyaml==5.1"
    ],
)