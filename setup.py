"accelpy setup module."

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def _setcfg():
    import os, json

    cfgdir = os.path.join(os.path.expanduser("~"), ".accelpy")
    libdir = os.path.join(cfgdir, "lib")
    cfgfile = os.path.join(cfgdir, "config")

    if not os.path.isdir(libdir):
        os.makedirs(libdir)

    config = {
        "libdir": libdir,
        "blddir": "",
        "session": {}
    }

    with open(cfgfile, "w")  as f:
        json.dump(config, f)


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        _setcfg()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        _setcfg()


def main():

    from accelpy.core import name, version, description, long_description, author

    install_requires = ["numpy"]
    console_scripts = ["accelpy=accelpy.command:main"]
    keywords = ["GPU", "CPU", "Accelerator", "Cuda", "Hip",
                "OpenAcc", "OpenMP", "Numpy", "C++", "Fortran", "accelpy"]

    setup(
        name=name,
        version=version,
        description=description,
        long_description=long_description,
        author=author,
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
        cmdclass={
            'develop': PostDevelopCommand,
            'install': PostInstallCommand,
        },
        keywords=keywords,
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
