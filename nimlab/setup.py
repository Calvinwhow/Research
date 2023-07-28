from setuptools import setup, find_packages

setup(
    name="nimlab",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        "nimlab/scripts/connectome_quick.py",
        "nimlab/scripts/target_roi_correl.py",
        "nimlab/scripts/convert_csv.py",
        "nimlab/scripts/calc_cort_thickness_glm_fs5.sh",
        "nimlab/scripts/calc_cort_thickness_wmap_fs5.sh",
    ],
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "datalad",
        "ipywidgets",
        "pymongo",
        "pybids",
        "tqdm",
        "numba",
        "numpy",
        "scipy",
        "termcolor",
        "scikit-learn",
        "pandas",
        "h5py",
        "python-pptx",
        "nilearn",
        "hdf5storage",
        "natsort",
        "matplotlib",
        "statsmodels",
        "fslpy",
        "PyYAML",
    ],
    package_data={
        "": [
            "*.yaml",
            "*.json",
            "*.sh",
            "*.nii",
            "*.nii.gz",
            "*.txt",
            "*.rst",
            "*.gii",
            "*.gii.gz",
        ],
        # And include any *.msg files found in the 'hello' package, too:
        "hello": ["*.msg"],
    },
    # metadata for upload to PyPI
    author="Alexander Cohen, Christopher Lin, William Drew, and Louis Soussand",
    author_email="alexander.cohen2@childrens.harvard.edu",
    description="nimlab internal package",
    license="MIT",
    keywords="neuroimaging connectome nimlab",
    url="http://github.com/nimlab/software_env/",  # project home page, if any
    project_urls={
        "Bug Tracker": "https://bugs.example.com/HelloWorld/",
        "Documentation": "https://docs.example.com/HelloWorld/",
        "Source Code": "https://code.example.com/HelloWorld/",
    },
)
