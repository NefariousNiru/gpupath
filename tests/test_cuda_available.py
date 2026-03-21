# file: tests/test_cuda_available.py

from gpupath import cuda_available, cuda_info


def test_cuda_available():
    assert cuda_available() == True


def test_cuda_info():
    assert cuda_info()["cuda_available"] == True
