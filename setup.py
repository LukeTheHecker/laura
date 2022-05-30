import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="laura",
    version="0.0.2",
    author="Lukas Hecker",
    author_email="lukas_hecker@web.de",
    description="**laura**: Local Auto-Regressive Average",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LukeTheHecker/laura",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'mne', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.0',
)
