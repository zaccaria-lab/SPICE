import setuptools
from setuptools import setup


setuptools.setup(
    name='spice',
    # Keep in sync with spice/__init__.py __version__.
    version='0.1.0',
    python_requires='>=3.9',
    packages=[
        'spice',
        'spice.core',
        'spice.core.optimal_tree',
        'spice.core.multi_solutions',
        'spice.core.bootstrapping',
        'spice.core.auxillary',
        'spice.io',
        'spice.sh',
        'spice.plotting',
        'spice.postprocess',
    ],
    author='Abigail Bunkum and Simone Zaccaria',
    author_email='s.zaccaria@ucl.ac.uk',
    description='Subclone Probability Inference of Copy-number Evolution (SPICE)',
    long_description='https://github.com/zaccaria-lab/SPICE',
    url='https://github.com/zaccaria-lab/SPICE',
    # Versions follow the tested environment in spice_env.yaml.
    # NOTE: Gurobi is a required solver but is provided via the conda recipe
    # (meta.yaml run requirements), not through pip, so it is intentionally
    # omitted from install_requires here.
    install_requires=[
        'numpy<2',
        'pandas>=2.0',
        'scipy>=1.10',
        'pyomo>=6.7',
        'biopython>=1.80',
        'matplotlib>=3.7',
        'seaborn>=0.13',
    ],
    license='BSD 3-Clause',
    platforms=["Linux", "MacOs", "Windows"],
    classifiers=[
        'Programming Language :: Python :: 3',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    include_package_data=False,
    zip_safe=True,
    keywords=[
        'scientific',
        'sequence analysis',
        'cancer',
        'DNA',
        'copy-number',
        'phylogenetics',
        'tumour evolution'],
    entry_points={'console_scripts': ['spice=spice.sh.spice:main']}
)
