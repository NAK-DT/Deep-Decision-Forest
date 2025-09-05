from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize
import platform
from pathlib import Path
# Read version without importing the package
def get_version():
    version_file = Path(__file__).parent / "pydl85" / "_version.py"
    if version_file.exists():
        with open(version_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip("\"'")
    return "1.0.0"  # fallback

__version__ = get_version()



DISTNAME = 'pydl8.5'
DESCRIPTION = 'A package to build an optimal binary decision tree classifier.'
this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "README.md").read_text()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
AUTHORS = 'Gael Aglin, Siegfried Nijssen, Pierre Schaus'
AUTHORS_EMAIL = 'aglingael@gmail.com, siegfried.nijssen@gmail.com, pschaus@gmail.com'
URL = 'https://github.com/aia-uclouvain/pydl8.5'
LICENSE = 'LICENSE.txt'
DOWNLOAD_URL = 'https://github.com/aia-uclouvain/pydl8.5'
VERSION = __version__
INSTALL_REQUIRES = ["setuptools", "cython", "numpy", "scikit-learn", "cvxpy"]
KEYWORDS = ['decision trees', 'discrete optimization', 'classification']
CLASSIFIERS = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
        'graphviz'
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'sphinxcontrib',
        'sphinx_copybutton',
        'matplotlib',
        'scipy',
        'pillow',
    ],
    'boosting': [
        "cvxpy"
    ]
}
PROJECT_URLS = {
    "Source on github": "https://github.com/aia-uclouvain/pydl8.5",
    "Documentation": "https://pydl85.readthedocs.io/en/latest/?badge=latest",
}

EXTENSION_BUILD_ARGS = ['/std:c++20', '/DCYTHON_PEP489_MULTI_PHASE_INIT=0'] if platform.system() == "Windows" else ['-std=c++20', '-DCYTHON_PEP489_MULTI_PHASE_INIT=0', '-fno-wrapv']
EXTENSION_INCLUDE_DIRS = ['core/src', 'cython_extension']
# Now we have the actual C++ implementation files
COMMON_CPP_SOURCES = [
    'core/src/cache.cpp',
    'core/src/cache_hash_cover.cpp',
    'core/src/cache_hash_itemset.cpp',
    'core/src/cache_trie.cpp',
    'core/src/dataManager.cpp',
    'core/src/depthTwoComputer.cpp',
    'core/src/dl85.cpp',
    'core/src/globals.cpp',
    'core/src/nodeDataManager.cpp',
    'core/src/nodeDataManager_Cover.cpp',
    'core/src/nodeDataManager_Trie.cpp',
    'core/src/rCover.cpp',
    'core/src/rCoverFreq.cpp',
    'core/src/rCoverWeight.cpp',
    'core/src/search_base.cpp',
    'core/src/search_cover_cache.cpp',
    'core/src/search_nocache.cpp',
    'core/src/search_trie_cache.cpp',
    'core/src/solution.cpp',
    'core/src/solution_Cover.cpp',
    'core/src/solution_Trie.cpp'
]

# Build single extension with only dl85Optimizer.pyx to avoid symbol conflicts
dl85_extension = Extension(
    name="cython_extension.dl85Optimizer",
    language="c++",
    sources=[
        "cython_extension/dl85Optimizer.pyx",
        # Remove error_function.pyx to avoid symbol conflicts
    ] + COMMON_CPP_SOURCES,  # Add the C++ sources
    include_dirs=EXTENSION_INCLUDE_DIRS,
    extra_compile_args=EXTENSION_BUILD_ARGS,
    extra_link_args=EXTENSION_BUILD_ARGS
)

# Comment out the separate error_function extension
# error_func_extension = Extension(
#     name="cython_extension.error_function",
#     language="c++",
#     sources=["cython_extension/error_function.pyx"],
#     include_dirs=EXTENSION_INCLUDE_DIRS,
#     extra_compile_args=EXTENSION_BUILD_ARGS,
#     extra_link_args=EXTENSION_BUILD_ARGS
# )

setup(
    name=DISTNAME,
    version=VERSION,
    url=URL,
    project_urls=PROJECT_URLS,
    author=AUTHORS,
    author_email=AUTHORS_EMAIL,
    download_url=DOWNLOAD_URL,
    license="MIT",  # or LICENSE.txt if it exists
    include_package_data=True,
    packages=find_packages(),
    package_data={"cython_extension": ["*.pxd"]},
    keywords=KEYWORDS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    zip_safe=True,
    ext_modules=cythonize(
        [dl85_extension],  # Only one extension now
        compiler_directives={
            "language_level": "3",
            "c_string_type": "unicode",
            "c_string_encoding": "utf8"
        },
    )
)