from setuptools import setup, find_packages

setup(
    name="amp_parkinsons",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'lightgbm>=3.3.0',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'pyyaml>=6.0'
    ],
    python_requires='>=3.8',
)