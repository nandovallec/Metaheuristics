from setuptools import setup
from Cython.Build import cythonize
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
setup(
    # ext_modules = cythonize("main.py", include_path=[numpy.get_include()])
    name = 'Test app',
    ext_modules = [Extension('test',sources=['main.py'],
                                   extra_compile_args=['-O3'],
                                   language='c++', include_path=[numpy.get_include()])
                     ],
                     cmdclass = {'build_ext': build_ext}
)