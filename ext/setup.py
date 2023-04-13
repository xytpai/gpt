from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gpt_ext',
    ext_modules=[
        CUDAExtension(
            'gpt_ext', 
            ['gpt_ext.cpp', 'rms_norm.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
