from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

base_packages = ["numpy>=1.16.2", "pandas>=0.24.2", "networkx>=2.2", "scikit-learn"]
docs_packages = ["black"]
test_packages = ["pytest", "ipython"]
dev_packages = (
    ["notebook", "ipywidgets", "matplotlib", "seaborn", "altair", "themes"]
    + docs_packages
    + test_packages
)


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
    install_requires=base_packages,
    extras_require={
        "docs": docs_packages,
        "test": test_packages,
        "dev": dev_packages,
    },
    include_package_data=True,
    # package_data={'datasets': ['netsci/resources/datasets/*']},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
