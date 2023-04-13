import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bts",
    version="0.1.0",
    author="Luca Zoffoli",
    author_email="lzoffoli@technogym.com",
    description="TDF File reader",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zof1985/bts",
    license="NO LICENCE",
    packages=["bts"],
    install_requires=["numpy", "pandas"],
    requires=["numpy", "pandas"],
)
