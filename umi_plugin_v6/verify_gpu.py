"""GPU 验证脚本：检查 ONNX Runtime CUDA 是否真正可用。
被 install_gpu.bat / install_gpu_rtx50.bat 调用，替代超长 python -c 命令
（避免 cmd.exe 括号解析错误）。退出码 0 = CUDA 可用，1 = CUDA 不可用。

判定逻辑（issue #10 / #15 教训）：
  get_available_providers() 只检查 EP 是否编译进 onnxruntime，不检查
  CUDA/cuDNN DLL 是否能真正加载（ORT 会静默回退 CPU）。
  本脚本的终极判定 = 直接加载 ORT 自带的 CUDA provider DLL：
  Windows 加载器沿 DLL 链接关系自动解析 cudart/cublas/cudnn 依赖，
  与 CUDA 大版本无关（12/13 均适用），加载成功即 CUDA EP 可创建。
  另用 glob 模式匹配（cudart64_*.dll）做诊断输出——
  硬编码 cudart64_12.dll 等文件名会在 CUDA 13 环境（onnxruntime-gpu
  1.27+ 为 CUDA 13 构建，配 cu13 系列 nvidia 包，DLL 名为
  cudart64_13.dll 等）误报失败（issue #15 根因）。
"""
import sys
import os
import sysconfig
import ctypes
import glob
import subprocess


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


def find_dlls(pattern):
    """在所有 nvidia DLL 目录中 glob 搜索，返回匹配的 DLL 文件名（小写、去重、排序）。

    版本无关：cudart64_*.dll 同时匹配 CUDA 12（cudart64_12.dll）、
    CUDA 13（cudart64_13.dll）及未来大版本。
    """
    hits = set()
    for dll_dir in _find_nvidia_dll_dirs():
        for path in glob.glob(os.path.join(dll_dir, pattern)):
            hits.add(os.path.basename(path).lower())
    return sorted(hits)


def try_load_ort_cuda_provider():
    """终极验证：直接加载 ORT 自带的 CUDA provider DLL（issue #15 修复核心）。

    onnxruntime_providers_cuda.dll 依赖 cudart/cublas/cudnn 等运行时 DLL，
    Windows 加载器会沿已注册的 DLL 目录解析整条依赖链：
      - 加载成功 == CUDAExecutionProvider 可正常创建（不会静默回退 CPU）
      - 加载失败 == 缺 DLL 或版本不匹配，ORT 必然回退 CPU
    该判定与 CUDA 大版本无关，不需要硬编码任何 DLL 文件名。
    返回 (成功与否, 失败原因)。
    """
    try:
        import onnxruntime as ort
        dll_path = os.path.join(
            os.path.dirname(ort.__file__), "capi", "onnxruntime_providers_cuda.dll"
        )
        if not os.path.exists(dll_path):
            return False, "onnxruntime_providers_cuda.dll not found (onnxruntime-gpu not installed?)"
        try:
            ctypes.WinDLL(dll_path)
            return True, None
        except OSError as e:
            return False, str(e)
    except Exception as e:
        return False, str(e)


def _ver_tuple(version):
    """'1.28.0' -> (1, 28)；忽略 dev/rc 等后缀（'1.28.0dev' -> (1, 28)）。"""
    parts = []
    for p in version.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) if len(parts) == 2 else (0, 0)


def get_gpu_info():
    """通过 nvidia-smi 获取 (GPU 名称, compute capability)。失败返回 (None, None)。

    compute capability >= 12.0 即 RTX 50 系（Blackwell sm_120）——
    需要 onnxruntime-gpu CUDA 13 构建（1.27+，PyPI 默认），旧版
    CUDA 12.8 构建（<=1.26）不含 sm_120 内核，会静默回退 CPU。
    """
    queries = (
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    )
    for cmd in queries:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
                name = parts[0] if parts else ""
                cap = parts[1] if len(parts) > 1 else ""
                if name:
                    return name, cap
        except Exception:
            pass
    return None, None


def main():
    setup_nvidia_dlls()
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.")
        sys.exit(1)

    providers = ort.get_available_providers()
    print("Available providers:", providers)
    print("onnxruntime version:", ort.__version__)

    if "CUDAExecutionProvider" not in providers:
        print("ERROR: CUDAExecutionProvider is NOT available.")
        print("onnxruntime-gpu may not be installed correctly.")
        sys.exit(1)

    # ---- GPU 信息 + RTX 50 系（Blackwell sm_120）检测 ----
    gpu_name, compute_cap = get_gpu_info()
    is_blackwell = False
    if gpu_name:
        cap_str = " (compute capability %s)" % compute_cap if compute_cap else ""
        print("GPU: %s%s" % (gpu_name, cap_str))
        if compute_cap:
            try:
                is_blackwell = float(compute_cap) >= 12.0
            except ValueError:
                pass
        if not is_blackwell and "RTX 50" in gpu_name.upper():
            is_blackwell = True

    # ---- 终极验证：加载 ORT CUDA provider DLL（版本无关）----
    print("")
    print("Loading onnxruntime CUDA provider DLL...")
    provider_ok, provider_err = try_load_ort_cuda_provider()

    # ---- 诊断：glob 扫描核心 CUDA/cuDNN DLL（版本无关）----
    print("Scanning CUDA/cuDNN DLLs (any CUDA major version)...")
    diag = {}
    for pattern, desc in (
        ("cudart64_*.dll", "CUDA Runtime"),
        ("cudnn64_*.dll", "cuDNN"),
        ("cublas64_*.dll", "cuBLAS"),
        ("cublasLt64_*.dll", "cuBLAS Lt"),
    ):
        found = find_dlls(pattern)
        diag[desc] = found
        if found:
            print("  [OK] %s: %s" % (desc, ", ".join(found)))
        else:
            print("  [--] %s: not found in pip nvidia packages" % desc)

    print("")
    if provider_ok:
        print("CUDA provider DLL loaded - all dependencies resolved!")
        if is_blackwell:
            print("")
            # onnxruntime-gpu 1.27+ on PyPI is built with CUDA 13 and
            # includes sm_120 kernels; <=1.26 (CUDA 12.8 build) does not.
            ver = _ver_tuple(ort.__version__)
            if ver >= (1, 27):
                print("[OK] RTX 50 series with onnxruntime-gpu %s" % ort.__version__)
                print("     (CUDA 13 build, includes sm_120 kernels).")
            else:
                print("[WARNING] RTX 50 series (Blackwell sm_120) detected with")
                print("          onnxruntime-gpu %s (CUDA 12.8 build)." % ort.__version__)
                print("          This build does NOT include sm_120 kernels,")
                print("          CUDA EP may silently fall back to CPU.")
                print("          Run install_gpu_rtx50.bat to install the CUDA 13")
                print("          build (requires NVIDIA driver R580+).")
        print("")
        print("GPU verification PASSED.")
        sys.exit(0)

    # ---- 失败：输出诊断与建议 ----
    print("ERROR: Failed to load onnxruntime CUDA provider DLL!")
    if provider_err:
        print("  Reason: %s" % provider_err)
    print("")
    print("This means onnxruntime-gpu is installed but CUDA/cuDNN runtime")
    print("DLLs cannot be loaded. ORT will silently fall back to CPU mode.")
    print("")
    missing = [d for d, f in diag.items() if not f]
    if missing:
        print("Missing DLL groups (in pip nvidia packages): " + ", ".join(missing))
        print("Common causes:")
        print("  1. CUDA/cuDNN pip packages failed to install - re-run the install script")
        print("  2. NVIDIA driver is too old - update to latest version")
        print("  3. Conflicting onnxruntime CPU version still installed")
    else:
        print("All expected CUDA/cuDNN DLLs were found, but the provider DLL")
        print("still failed to load. This usually means a version mismatch")
        print("between onnxruntime-gpu and the installed CUDA/cuDNN packages")
        print("(e.g. CUDA 12 build of ORT with CUDA 13 runtime DLLs, or vice versa).")
        print("Re-run the install script to get a matching set.")
    print("")
    print("NVIDIA driver download: https://www.nvidia.com/Download/index.aspx")
    sys.exit(1)


if __name__ == "__main__":
    main()
