import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rank_greville",
    version="1.0",
    author="Ruben Staub",
    author_email="ruben.staub@ens-lyon.fr",
    description="Rank-decomposition based recursive linear solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RubenStaub/rank-greville",
#    packages=setuptools.find_packages(),
    py_modules=['rank_greville'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
#    python_requires='>=3.5', # python2 support is left to the motivated user...
    install_requires=['numpy']
)

