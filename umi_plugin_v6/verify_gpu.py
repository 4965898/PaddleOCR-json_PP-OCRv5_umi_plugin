"""GPU verification script: check if ONNX Runtime CUDA is truly available.
Called by install_gpu.bat to replace the long python -c command
(avoids cmd.exe parenthesis parsing errors).
Exit code 0 = CUDA available, 1 = CUDA not available.

Key: get_available_providers() only checks if EP is compiled into onnxruntime,
not whether CUDA/cuDNN DLLs can actually be loaded. This script uses ctypes
to load cuDNN DLLs directly, detecting silent CPU fallback (issue #10 root cause).
"""
import sys
import os
import sysconfig
import ctypes


def setup_nvidia_dlls():
    """Add pip-installed nvidia CUDA/cuDNN DLL paths to search path."""
    try:
        site_dir = sysconfig.get_paths()["purelib"]
        nvidia_base = os.path.join(site_dir, "nvidia")
        if not os.path.isdir(nvidia_base):
            return
        for sub in os.listdir(nvidia_base):
            dll_dir = os.path.join(nvidia_base, sub, "bin")
            if os.path.isdir(dll_dir):
                try:
                    os.add_dll_directory(dll_dir)
                except Exception:
                    pass
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def find_nvidia_bin_dirs():
    """Return list of all pip-installed nvidia */bin directories."""
    dirs = []
    try:
        site_dir = sysconfig.get_paths()["purelib"]
        nvidia_base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_base):
            for sub in os.listdir(nvidia_base):
                dll_dir = os.path.join(nvidia_base, sub, "bin")
                if os.path.isdir(dll_dir):
                    dirs.append(dll_dir)
    except Exception:
        pass
    return dirs


def try_load_dll(dll_name):
    """Try to load a DLL, return True on success, False on failure."""
    try:
        ctypes.WinDLL(dll_name)
        return True
    except OSError:
        pass
    for dll_dir in find_nvidia_bin_dirs():
        dll_path = os.path.join(dll_dir, dll_name)
        if os.path.exists(dll_path):
            try:
                ctypes.WinDLL(dll_path)
                return True
            except OSError:
                pass
    return False


def main():
    setup_nvidia_dlls()
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.")
        sys.exit(1)

    providers = ort.get_available_providers()
    print("Available providers:", providers)

    if "CUDAExecutionProvider" not in providers:
        print("ERROR: CUDAExecutionProvider is NOT available.")
        print("onnxruntime-gpu may not be installed correctly.")
        sys.exit(1)

    # Critical verification: try loading CUDA/cuDNN DLLs
    # get_available_providers() only checks compile-time support,
    # not runtime DLL availability. ORT silently falls back to CPU
    # if cuDNN cannot be loaded at session creation (issue #10 root cause).
    print("")
    print("Verifying CUDA/cuDNN DLLs...")

    dll_checks = [
        ("cudart64_12.dll", "CUDA Runtime"),
        ("cudnn64_9.dll", "cuDNN 9"),
        ("cudnn64_8.dll", "cuDNN 8"),
        ("cublas64_12.dll", "cuBLAS"),
        ("cublasLt64_12.dll", "cuBLAS Light"),
    ]

    loaded_dlls = []
    for dll_name, dll_desc in dll_checks:
        if try_load_dll(dll_name):
            loaded_dlls.append(f"{dll_desc} ({dll_name})")
            print(f"  [OK] {dll_desc}: {dll_name}")
        else:
            if dll_name == "cudnn64_8.dll" and any("cudnn64_9" in d for d in loaded_dlls):
                continue
            if dll_name == "cublasLt64_12.dll":
                print(f"  [SKIP] {dll_desc}: {dll_name} (optional)")
                continue
            print(f"  [FAIL] {dll_desc}: {dll_name} NOT FOUND")

    print("")

    has_cudart = any("CUDA Runtime" in d for d in loaded_dlls)
    has_cudnn = any("cuDNN" in d for d in loaded_dlls)
    has_cublas = any("cuBLAS" in d and "Light" not in d for d in loaded_dlls)

    if has_cudart and has_cudnn and has_cublas:
        print("All critical CUDA/cuDNN DLLs loaded successfully!")
        print("GPU verification PASSED.")
        sys.exit(0)
    else:
        print("ERROR: Some critical CUDA/cuDNN DLLs failed to load!")
        print("")
        print("This means onnxruntime-gpu is installed but CUDA/cuDNN")
        print("runtime DLLs cannot be loaded. ORT will silently fall")
        print("back to CPU mode (GPU will NOT be used).")
        print("")
        print("Common causes:")
        print("  1. NVIDIA driver is too old - update to latest version")
        print("  2. CUDA/cuDNN packages failed to install - re-run install_gpu.bat")
        print("  3. Conflicting onnxruntime CPU version still installed")
        print("")
        print("NVIDIA driver download: https://www.nvidia.com/Download/index.aspx")
        sys.exit(1)


if __name__ == "__main__":
    main()
