"""GPU 验证脚本：检查 ONNX Runtime CUDA 是否真正可用。
被 install_gpu.bat 调用，替代超长 python -c 命令（避免 cmd.exe 括号解析错误）。
退出码 0 = CUDA 可用，1 = CUDA 不可用。

关键：get_available_providers() 只检查 EP 是否编译进 onnxruntime，
不检查 CUDA/cuDNN DLL 是否能真正加载。此处通过 ctypes 加载 cuDNN DLL
来验证运行时可用性，避免 ORT 静默回退 CPU（issue #10 根因）。
"""
import sys
import os
import sysconfig
import ctypes
import glob


def _find_nvidia_dll_dirs():
    """递归扫描 nvidia\\ 下所有子目录，返回包含 .dll 文件的目录列表。

    兼容不同版本 nvidia pip 包的目录结构差异（issue #10 根因）：
      - 正常结构：nvidia\\<sub>\\bin\\*.dll
      - cu13 包结构：nvidia\\cu13\\bin\\x86_64\\*.dll
      - 旧版 cublas：nvidia\\cublas\\*.dll（无 bin 子目录）
    """
    dirs = []
    try:
        site_dir = sysconfig.get_paths()["purelib"]
        nvidia_base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_base):
            for root, _subdirs, files in os.walk(nvidia_base):
                if any(f.lower().endswith(".dll") for f in files):
                    if root not in dirs:
                        dirs.append(root)
    except Exception:
        pass
    return dirs


def setup_nvidia_dlls():
    """将 pip 安装的 nvidia CUDA/cuDNN DLL 路径加入搜索路径。

    递归扫描所有包含 .dll 的子目录（不局限于 <sub>\\bin\\），
    确保不同版本 nvidia pip 包的 DLL 都能被找到（issue #10 根因）。
    """
    try:
        for dll_dir in _find_nvidia_dll_dirs():
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def find_nvidia_bin_dirs():
    """返回 pip 安装的所有包含 nvidia DLL 的目录列表（向后兼容）。"""
    return _find_nvidia_dll_dirs()


def try_load_dll(dll_name):
    """尝试加载指定 DLL，成功返回 True，失败返回 False。"""
    # 先尝试系统搜索路径（已由 setup_nvidia_dlls 设置）
    try:
        ctypes.WinDLL(dll_name)
        return True
    except OSError:
        pass
    # 逐个 nvidia bin 目录尝试
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

    # 关键验证：尝试加载 CUDA/cuDNN DLL
    # get_available_providers() 只检查编译时支持，不检查运行时 DLL 可用性。
    # 只有真正加载 DLL 时才会发现 cuDNN 缺失等问题。
    # ORT 在创建 session 时如果无法加载 cuDNN，会静默回退到 CPU（issue #10 根因）。
    print("")
    print("Verifying CUDA/cuDNN DLLs...")

    # 检查关键 DLL
    # cudnn64_9.dll (cuDNN 9.x) 或 cudnn64_8.dll (cuDNN 8.x)
    # cudart64_12.dll (CUDA 12.x Runtime)
    # cublas64_12.dll (cuBLAS)
    dll_checks = [
        ("cudart64_12.dll", "CUDA Runtime"),
        ("cudnn64_9.dll", "cuDNN 9"),
        ("cudnn64_8.dll", "cuDNN 8"),
        ("cublas64_12.dll", "cuBLAS"),
        ("cublasLt64_12.dll", "cuBLAS Light"),
    ]

    all_loaded = True
    loaded_dlls = []
    for dll_name, dll_desc in dll_checks:
        if try_load_dll(dll_name):
            loaded_dlls.append(f"{dll_desc} ({dll_name})")
            print(f"  [OK] {dll_desc}: {dll_name}")
        else:
            # cudnn64_8 是可选的（cuDNN 9 已安装时不需要 8）
            if dll_name == "cudnn64_8.dll" and any("cudnn64_9" in d for d in loaded_dlls):
                continue
            # cublasLt 是可选的
            if dll_name == "cublasLt64_12.dll":
                print(f"  [SKIP] {dll_desc}: {dll_name} (optional)")
                continue
            print(f"  [FAIL] {dll_desc}: {dll_name} NOT FOUND")
            all_loaded = False

    print("")

    # 至少需要 CUDA Runtime + cuDNN + cuBLAS
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
