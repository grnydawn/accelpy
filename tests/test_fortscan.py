import os
from accelpy.fortscan import get_firstexec

fortexts = ['.f', '.f90', '.f95', '.f03', '.f08', '.for', '.ftn']

def ttest_e3sm():
    e3smpath = "/Users/8yk/repos/github/E3SM/components"

    for dirpath, dirnames, filenames in os.walk(e3smpath):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)

            if ext.lower() in fortexts:
                filepath = os.path.join(dirpath, filename)

                with open(filepath) as f:
                    print(filepath)
                    firstexec = get_firstexec(f.readlines())
                    print(str(firstexec))
