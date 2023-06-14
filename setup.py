import setuptools

setuptools.setup(
    name="filmfinder",
    version="1.0.0",
    author="Padchara Bubphasan",
    author_email="padchara.bubphasan@gmail.com",
    description="A short description of your package",
    long_description="long_description",
    long_description_content_type="text/markdown",
    url="https://github.com/icebub/filmfinder",
    packages=setuptools.find_packages("."),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
