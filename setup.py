"accelpy setup module."

def main():

    from setuptools import setup, find_packages

    install_requires = ["numpy"]
    console_scripts = ["accelpy=accelpy.__main__:main"]

    setup(
        name="accelpy",
        version="0.3.0",
        description="Scalable Accelerator Interface in Python",
        long_description="Scalable Accelerator Interface in Python",
        author="Youngsung Kim",
        author_email="youngsung.kim.act2@gmail.com",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        keywords="accelpy",
        packages=find_packages(exclude=["tests"]),
        include_package_data=True,
        install_requires=install_requires,
        entry_points={ "console_scripts": console_scripts },
        project_urls={
            "Bug Reports": "https://github.com/grnydawn/accelpy/issues",
            "Source": "https://github.com/grnydawn/accelpy",
        }
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
