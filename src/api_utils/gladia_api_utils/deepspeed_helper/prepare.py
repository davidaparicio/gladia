import os

GLADIA_PERSISTENT_PATH = os.getenv("GLADIA_PERSISTENT_PATH", "/gladia")
MAMBA_ROOT_PREFIX = os.getenv("MAMBA_ROOT_PREFIX", f"{GLADIA_PERSISTENT_PATH}/conda")


def prepare(*_, **__) -> None:

    PATH_TO_ENV = (
        f"{MAMBA_ROOT_PREFIX}/envs/deepspeed-mii"  # TODO: retrieve from kwargs
    )
    PYTHON_VERSION = "3.8"  # TODO: retrieve from kwargs

    FILE_PATH = f"{PATH_TO_ENV}/lib/python{PYTHON_VERSION}/site-packages/torch/utils/cpp_extension.py"

    cpp_extension = open(
        FILE_PATH,
    ).read()

    cpp_extension = cpp_extension.replace(
        """\
from .hipify import hipify_python
from .hipify.hipify_python import GeneratedFileCleaner
""",
        """\
try:
    from .hipify import hipify_python
    from .hipify.hipify_python import GeneratedFileCleaner
except:
    pass
""",
    )

    cpp_extension = cpp_extension.replace(
        """\
                with GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                    if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                        hipify_result = hipify_python.hipify(
                            project_directory=build_directory,
                            output_directory=build_directory,
                            header_include_dirs=(extra_include_paths if extra_include_paths is not None else []),
                            extra_files=[os.path.abspath(s) for s in sources],
                            ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')],  # no need to hipify ROCm or PyTorch headers
                            show_detailed=verbose,
                            show_progress=verbose,
                            is_pytorch_extension=True,
                            clean_ctx=clean_ctx
                        )

                        hipified_sources = set()
                        for source in sources:
                            s_abs = os.path.abspath(source)
                            hipified_sources.add(hipify_result[s_abs]["hipified_path"] if s_abs in hipify_result else s_abs)

                        sources = list(hipified_sources)

                    _write_ninja_file_and_build_library(
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags or [],
                        extra_cuda_cflags=extra_cuda_cflags or [],
                        extra_ldflags=extra_ldflags or [],
                        extra_include_paths=extra_include_paths or [],
                        build_directory=build_directory,
                        verbose=verbose,
                        with_cuda=with_cuda,
                        is_standalone=is_standalone)
""",
        """\
                if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                    hipify_result = hipify_python.hipify(
                        project_directory=build_directory,
                        output_directory=build_directory,
                        header_include_dirs=(extra_include_paths if extra_include_paths is not None else []),
                        extra_files=[os.path.abspath(s) for s in sources],
                        ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')],  # no need to hipify ROCm or PyTorch headers
                        show_detailed=verbose,
                        show_progress=verbose,
                        is_pytorch_extension=True,
                        clean_ctx=clean_ctx
                    )

                    hipified_sources = set()
                    for source in sources:
                        s_abs = os.path.abspath(source)
                        hipified_sources.add(hipify_result[s_abs]["hipified_path"] if s_abs in hipify_result else s_abs)

                    sources = list(hipified_sources)

                _write_ninja_file_and_build_library(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags or [],
                    extra_cuda_cflags=extra_cuda_cflags or [],
                    extra_ldflags=extra_ldflags or [],
                    extra_include_paths=extra_include_paths or [],
                    build_directory=build_directory,
                    verbose=verbose,
                    with_cuda=with_cuda,
                    is_standalone=is_standalone)
""",
    )

    open(
        FILE_PATH,
        "w",
    ).write(cpp_extension)
