"accelpy setup module."

import os, json
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

genv = {'__builtins__' : None}
consts = {}

with open(os.path.join("accelpy", "const.py")) as fp:
    exec(fp.read(), genv, consts)

print(consts)

def _setcfg():

    cfgdir = os.path.join(os.path.expanduser("~"), ".accelpy")

    libdir = os.path.join(cfgdir, "lib")
    cfgfile = os.path.join(cfgdir, "config")

    for vendor in consts["vendors"]:
        vendor_path = os.path.join(libdir, vendor)
        if not os.path.isdir(vendor_path):
            try:
                os.makedirs(vendor_path)

            except FileExistsError:
                pass

    config = {
        "libdir": libdir,
        "blddir": "",
    }

    if not os.path.isfile(cfgfile):
        with open(cfgfile, "w")  as f:
            json.dump(config, f)


class DevelopCommand(develop):
    def run(self):
        _setcfg()
        develop.run(self)


class InstallCommand(install):
    def run(self):
        _setcfg()
        install.run(self)


def main():

    install_requires = ["numpy"]
    console_scripts = ["accelpy=accelpy.command:main"]
    keywords = ["GPU", "CPU", "Accelerator", "Cuda", "Hip",
                "OpenAcc", "OpenMP", "Numpy", "C++", "Fortran", "accelpy"]

    setup(
        name=consts["name"],
        version=consts["version"],
        description=consts["description"],
        long_description=consts["long_description"],
        author=consts["author"],
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
            'develop': DevelopCommand,
            'install': InstallCommand,
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
