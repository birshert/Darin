from setuptools import setup, find_packages

setup(
    name="Renju game",
    description="Playing renju game vs computer",
    url="https://github.com/birshert/Darin",
    author="Birshert Alexey",
    classifiers=["Programming Language :: Python :: 3.7"],
    packages=find_packages(exclude=["JOUST", "KaggleFight", "NET", "labs_"]),
    python_requires=">=3.5, <4",
    install_requires=["numpy", "pygame", "torch"]
)
