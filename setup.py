import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name = "beam_simulator",
    version = "1.0",
    description = "Beam Simulator",
    long_description=long_description,
    author = "Christopher DiLullo",
    author_email = "cdilullo@unm.edu",
    url = "https://github.com/cdilullo/beam_simulator",
    license="GPL",
    classifiers = ['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Topic :: Scientific/Engineering:: Astronomy',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: POSIX :: Linux'],
    packages = find_packages(),
    python_requires = '>=3.6',
    install_requires = ['aipy>=3.0.1', 'numpy>=1.19', 'scipy>=1.5.4', 'astropy>=4.1',
                        'matplotlib>=3.3.2','numba>=0.51.2']
    )
