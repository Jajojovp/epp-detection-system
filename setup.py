from setuptools import setup, find_packages

setup(
    name="epp_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements/base.txt")
    ],
    author="Tu Nombre",
    author_email="tu@email.com",
    description="Sistema de detección de EPPs usando YOLOv5",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/epp-detection-system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
