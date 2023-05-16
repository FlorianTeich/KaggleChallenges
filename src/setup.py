import setuptools

setuptools.setup(
    name="kcu",
    version="0.0.1",
    author="Florian Teich",
    author_email="florianteich@gmail.com",
    description="Kaggle Challenge utils",
    long_description="Kaggle Challenge utils",
    long_description_content_type="text/markdown",
    url="https://github.com/FlorianTeich/KaggleChallenges",
    project_urls={
        "Bug Tracker": "https://github.com/FlorianTeich/KaggleChallenges/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"kcu": "kcu"},
    packages=["kcu"],
    python_requires=">=3.6",
)
