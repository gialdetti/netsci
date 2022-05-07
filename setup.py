from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.16.2", "pandas>=0.24.2", "networkx>=2.2", "scikit-learn"]

setup(
    name="netsci",
    version="0.0.3",
    author="Eyal Gal, Idan Segev, Michael London",
    author_email="eyalgl@gmail.com",
    description="Analyzing Complex Networks with Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/gialdetti/netsci/",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    # package_data={'datasets': ['netsci/resources/datasets/*']},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

