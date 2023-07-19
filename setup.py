"""The setup script."""

from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf8") as f:
    requirements = f.read().split("\n")

test_requirements = [
    "pytest>=3",
]

setup(
    author="QuandaGo Conversational Analytics",
    python_requires="==3.9.*",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9.*",
    ],
    install_requires=requirements,
    name="interactive_tm",
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    include_package_data=True,
    packages=find_packages(include=["assets", "assets.*, data., data.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    version="0.1.0",
    zip_safe=False,
)
