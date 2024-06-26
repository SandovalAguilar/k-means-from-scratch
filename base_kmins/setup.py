from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'K-Means algorithm from scratch.'
LONG_DESCRIPTION = 'K-Means algorithm from scratch.'

# Setting up
setup(
    name="kmins",
    version=VERSION,
    author="Ismael Sandoval, Mauricio Ponce",
    author_email="ismael.sandoval.aguilar@gmail.com, maupon2004@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scikit-learn'],
    keywords=['python', 'k-means', 'clustering'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)