from setuptools import setup, find_packages

setup(
    name="formula-ufmg-data-analysis",
    version="1.0.0",
    description="Data Analysis @ Horeb Energy Formula UFMG",
    author="Horeb Energy Formula UFMG",
    author_email="cachafeli@gmail.com",
    packages=find_packages(),
    install_requires=[
        "Flask>=2.0.0",
        "Flask-CORS>=3.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "plotly>=5.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "gunicorn>=20.0.0",
    ],
    python_requires=">=3.8",
)
