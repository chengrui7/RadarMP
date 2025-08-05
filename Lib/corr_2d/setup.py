from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

# get all .cu and .cpp files
sources = glob.glob('src/*.cpp') + glob.glob('src/*.cu')

setup(
    name='corr2d_cuda',  # module name
    version='0.1.0',     # version
    description='2D Correlation CUDA Operations',  # description
    ext_modules=[
        CUDAExtension(
            name='corr2d_cuda',  # modules
            sources=sources,     # source files
            extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)