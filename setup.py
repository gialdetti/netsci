from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.16.2", "pandas>=0.24.2", "matplotlib>=3.0.3", "seaborn>=0.9.0", "networkx>=2.2"]

setup(
    name="netsci",
    version="0.0.1",
    author="Eyal Gal",
    author_email="eyalgl@gmail.com",
    description="Analyzing Complex Networks with Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/gialdetti/netsci/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

