import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

LICENSE = 'MIT'

with open('requirements.txt') as f:
    requires = f.read().splitlines()

setuptools.setup(
    name="transferlearning",
    version="0.0.3",
    author="Song Cheng",
    author_email="chsong513@gmail.com",
    description="The integration of some popular transferlearning learning methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chsong513/TransferLearning",
    packages=setuptools.find_packages(),
    platforms=['all'],
    install_requires=requires,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)