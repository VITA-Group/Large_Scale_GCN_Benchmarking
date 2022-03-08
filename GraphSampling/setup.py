from setuptools import setup, find_packages
from torch.utils import cpp_extension


setup(name='GraphSampling',
      ext_modules=[
          cpp_extension.CppExtension(
              'cpp_extension.sample',
              ['./cpp_extension/sample.cpp']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)
