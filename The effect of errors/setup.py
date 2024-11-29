from setuptools import setup, find_packages

setup(
    name="trading_research",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pytest',
        'matplotlib',
        'seaborn',
    ]
)