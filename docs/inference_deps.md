# 推理依赖与测速（GPU 已打通，未装任何东西）

阶段 3 前的前置项：视觉推理原来 **954 ms/帧（约 1 fps）**，离实时差两个数量级。
结论先给：**GPU 上现状是 55 ms 车级 + 3~17 ms 装甲，整帧 ≈ 64~74 ms（14~15 fps）**，
落在「视觉可用」档。下面记事实、数字和怎么复现。

## 1. 环境事实（2026-11-21 实测）

| 项 | 实测值 |
| --- | --- |
| GPU | RTX 4060 Laptop（Max-Q），8 GB，驱动 595.99.02（支持 CUDA 13.2） |
| CUDA Toolkit | `/usr/local/cuda` 12.6.3，`libcublas.so.12` 等已在 `ldconfig` |
| CPU | Ryzen 7 8845H（8 核 16 线程），30 GB RAM，`/home` 剩 74 GB |
| onnxruntime-gpu | 1.26.0，providers = `Tensorrt / CUDA / CPU` |
| cuDNN 9 | **在用户 site**：`~/.local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9` |
| torch | **在用户 site** 2.10.0+cu128（torchvision 0.25.0+cu128 在 venv） |
| onnx / onnxscript | **都没有**（只在导出数字模型时需要，见第 4 节） |

这两样都在用户 site，而全仓库默认 `PYTHONNOUSERSITE=1`（避开用户 site 的 numpy 2.5.5
与 GNU Radio/OpenCV 的 ABI 冲突）——所以它们对 Python 不可见，但**库文件在磁盘上**。

## 2. 为什么以前会静默退 CPU

`onnxruntime-gpu` 找不到 `libcudnn.so.9` 时**不抛异常**：CUDA 执行器建失败，退到 CPU
照常返回数字。旧代码用 `importlib.util.find_spec("nvidia.cudnn")` 去找 cuDNN——在
`PYTHONNOUSERSITE=1` 下用户 site 不在 `sys.path` 上，`find_spec` 返回 `None`，
于是"库明明在磁盘上"被判成"没装"，帧率掉到 1fps 也没有任何报错。

顺带坑：`tools/bench_inference.py` 第一次跑 `--providers cuda` 时，ORT 静默退 CPU
返回了 46.7ms，我差点当成 GPU 数。现在工具会打 `fell_back_to_cpu: true` 并返回退出码 1。

## 3. 修法：按路径找，交给 `preload_dlls`（**不需要 sudo、不需要改环境变量**）

`jg_radar/integration/gpu_env.py` 扫两处 `site-packages` 下的 `nvidia/*/lib`：

1. venv：`<venv>/lib/python3.12/site-packages/nvidia/*/lib`（`pip install nvidia-cudnn-cu12` 的装法）
2. 用户 site：`~/.local/lib/python3.12/site-packages/nvidia/*/lib`（本机现状）

找到含 `libcudnn.so.9` 的目录后调用 `ort.preload_dlls(cuda=True, cudnn=True, directory=...)`。
实测这样就够——**不用 `LD_LIBRARY_PATH`、不用 re-exec、不用 apt**。已接到：
`OnnxBackend.load()`（所有视觉入口都走它）和 `tools/bench_inference.py`。
测试：`tests/test_gpu_env.py`（11 条）+ `tests/test_onnx_runtime_loading.py`（4 条）。

## 4. 实测数字

单模型（`tools/bench_inference.py`，dummy tensor，仅作加速比参考）：

| 模型 | CPU（4 线程最优） | GPU | 加速比 |
| --- | --- | --- | --- |
| 车级 1280×1280 | 675~1005 ms | **48~55 ms** | ≈14× |
| 装甲 192×192 | 22.5 ms | **5.2 ms** | ≈4× |

端到端（`--e2e`，走仓库真实级联：letterbox + NMS + ROI 批处理；1280×1024 帧，
办公室画面 0 检测，装甲级用人工 ROI 位形驱动——**这是测耗时，不是精度**）：

| 装甲 ROI 数 | 车级 ms | 装甲 ms | 合计 ms | fps |
| --- | --- | --- | --- | --- |
| 1 | 56.2 | 8.4 | 64.6 | 15.5 |
| 4 | 56.4 | 7.3 | 63.8 | 15.7 |
| 6 | 52.6 | 10.7 | 63.3 | 15.8 |
| 10 | 56.5 | 17.1 | 73.6 | 13.6 |
| 16 | 56.5 | 28.1 | 84.6 | 11.8 |
| 32 | 51.8 | 54.6 | 106.4 | 9.4 |

**装甲级是批处理，不是线性叠加**：6→10 个 ROI 只多 6 ms（按 5.4ms/ROI 线性算会多 22ms）。
所以"每帧最多 6~10 个 ROI"这条经验上限**偏保守**——实测到 32 个 ROI 仍不到 110 ms。
真机上 ROI 数由车框决定，不会长成 32；保留上限是为了给数字级和预处理留余量。

## 5. 数字分类（待办）：`.pth` → ONNX 被一个小依赖挡住

`tools/export_digit_onnx.py` 已写好（含"权重键名带 `backbone.` 前缀需要剥一层"的处理，
这份 MobileNetV2.2.pth 就是带前缀的）。torch 在用户 site 可见（去掉 `PYTHONNOUSERSITE`
即可），但导出还缺序列化图用的包：

- `onnx` **8.5 MB**（TorchScript 导出器），或
- `onnxscript` **0.7 MB**（dynamo 导出器）

**没有安装**，等拍板。装完一条命令：

```zsh
PYTHONNOUSERSITE= .venv/bin/python tools/export_digit_onnx.py \
    --weights /home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/weights/MobileNetv2.2.pth \
    --out models/digit_mobilenetv2.onnx
```

导出后运行时就只依赖 onnxruntime；不需要为了让 torch 可见而关掉 `PYTHONNOUSERSITE`
（那会把用户 site 的 numpy 2.5.5 带进来，GNU Radio/OpenCV 会崩）。

## 6. 复现命令

```zsh
# 单模型：CPU 与 GPU 各跑一遍
.venv/bin/python tools/bench_inference.py --config config/vision_check.yaml --providers cpu --threads 4
.venv/bin/python tools/bench_inference.py --config config/vision_check.yaml --providers auto

# 端到端（真实级联路径），结果落 out/
.venv/bin/python tools/bench_inference.py --e2e --config config/vision_check.yaml \
    --video /tmp/uvc_30fps.mp4 --resize 1280x1024 --rois 6 --out out/gpu_bench_20261121
```

## 7. 结论边界

- 这些是**性能测量**，不是视觉精度证据，不进阶段 3 验收表。
- 帧来源是内置 UVC（办公室画面），检测数是 0；装甲级耗时为人工 ROI 驱动。
  真机上海康相机、真实赛场画面下的数字可能不同，**到时要重量**。
- 数字分类仍未接入（等 `onnx` 或 `onnxscript`）；在那之前身份识别只能靠配色和规则，
  真实链路联调前必须补上。
