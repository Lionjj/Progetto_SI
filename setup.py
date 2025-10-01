from setuptools import setup, find_packages

setup(
    name="", # Nome pacchetto
    version="0.0.1", # Versione 
    author="Danilo Ivone",
    author_email="danilo.ivone.18@gmail.com",
    description="Descrizione del progetto",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="", # Repo github
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS indipendent",
    ],
    python_requires=">= 3.7",
)