from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="morphology_gnn",
    version="0.1.0",
    description="GNN for molecular property prediction in morphology.",
    author="clint",
    author_email="clinthoeselvan@hotmail.com",
    packages=find_packages(include=["morphology_gnn", "morphology_gnn.*"]),
    package_data={
        "morphology_gnn.cuda_radius_graph": [
            "radius_graph.cpp",
            "radius_graph_kernel.cu",
        ],
    },
    install_requires=[
        "torch>=2.6.0",
        "torch-geometric>=2.6.0",
        "scipy>=1.16.3",
        "numpy>=2.2.6",
    ],
    ext_modules=[
        CUDAExtension(
            name="morphology_gnn.cuda_radius_graph._cuda_radius_graph",
            sources=[
                "morphology_gnn/cuda_radius_graph/radius_graph.cpp",
                "morphology_gnn/cuda_radius_graph/radius_graph_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
