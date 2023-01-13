from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="torchvision_cpu_dcn",
    ext_modules=[
        CppExtension(
            "torchvision_cpu_dcn",
            ["cpu_dcn_kernel.cpp"],
            libraries=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
