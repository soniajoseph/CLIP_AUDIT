from setuptools import setup, find_packages

setup(
    name="clip_audit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # For example:
        # "numpy>=1.18.0",
        # "torch>=1.7.0",
    ],
    author="Sonia Joseph",
    author_email="your.email@example.com",
    description="A package for collecting activation intervals",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/soniajoseph/CLIP_AUDIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)