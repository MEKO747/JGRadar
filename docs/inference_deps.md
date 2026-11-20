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

## 5. 数字分类：已切成 ONNX 运行时（不再依赖 torch）

`.pth` 已导出成 ONNX，运行时走 `OnnxDigitBackend`（onnxruntime），**不需要 torch**：

```zsh
# 导出（一次性，需要 torch+torchvision+onnx；torch 在用户 site，所以这次要去掉 PYTHONNOUSERSITE）
PYTHONNOUSERSITE= .venv/bin/python tools/export_digit_onnx.py \
    --weights /home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/weights/MobileNetv2.2.pth \
    --out   /home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/weights/digit/digit_mobilenetv2.onnx
```

- 产物 8.49 MB，`torch 2.10.0+cu128`，MobileNetV2 / 64px / 6 类
- 这份 `.pth` 的键名带 `backbone.` 前缀，导出脚本里会剥一层（不剥就是一大串 missing key）
- `torch.onnx.export` 默认走 dynamo 导出器（要 onnxscript），脚本显式 `dynamo=False` 用 TorchScript 导出器，只依赖 `onnx`
- `config/vision_check.yaml` 已改成 `digit_backend: onnx` + `digit_model: .../digit.onnx`；
  `ega_replay.yaml` 仍是 `mobilenet`，两条路都保留

### 安装踩的坑（值得记下来）

`pip install onnx` 默认拉 **onnx 1.23.2**，它会连带把 venv 里的 **numpy 升到 2.5.5**，
于是 scipy 1.11.4（要 numpy<1.28）和 cv2 的 ABI 全被打坏。处理：

```zsh
pip install "onnx==1.18.0" "numpy==1.26.4"   # 1.18 只要求 numpy>=1.22，与 1.26 兼容
```

现在的组合：`numpy 1.26.4 / onnx 1.18.0 / cv2 4.6.0 / scipy 1.11.4`，
`import numpy, onnx, cv2, scipy` 全部正常，全量 pytest `806 passed, 1 skipped`。
**以后往这个 venv 装包，装完必须验 numpy 版本没被顶上去。**

## 5.1 证据自证：`providers` 列表 ≠ 真在 GPU 上算

`session.get_providers()` 只说"执行器建起来了"。CUDA 执行器能建起来、却一个节点都
没分到时，它**照样排在列表最前面**——所以 JSON 里写着 CUDA 不等于真在 GPU 上跑。

`tools/bench_inference.py --e2e` 现在额外做一次 ORT profiling（一次性，不计入计时），
统计**每个 provider 实际分到的节点数**，写进 JSON：

```json
"provider_nodes": {"car":   {"CUDAExecutionProvider": 398},
                   "armor":  {"CUDAExecutionProvider": 520, "CPUExecutionProvider": 201},
                   "digit":  {"CUDAExecutionProvider": 100}},
"cuda_nodes": 1018, "cpu_nodes": 201, "fell_back_to_cpu": false
```

- **CUDA 节点数为 0 时 bench 返回退出码 1**：宁可跑失败，也不交一个看不出真假的数。
- 顺手发现一个事实：**装甲模型有 201 个节点落在 CPU**（520 在 GPU）。整帧 10.6ms 里
  包含这部分，属于当前真实状态，不是测量误差。
- 单模型路径与 e2e 路径现在同口径都有 `fell_back_to_cpu`。

## 5.2 装甲模型那 201 个 CPU 节点：根因是 **batch 动态**，不是 H/W

`provider_nodes` 报出装甲模型 520 个节点在 GPU、201 个在 CPU，全部落在
`/model.6/.../attn/` 的 `Gather/Unsqueeze/Slice/Concat` 形状算子里。做了三组对照
（同一份 bench、同一条级联、同样的 6 / 10 个 ROI）：

| 模型变体 | 输入 | armor 节点 | 6 ROI | 10 ROI |
| --- | --- | --- | --- | --- |
| 现状 `armor_best.onnx` | `batch×3×high×wide`（全动态） | CUDA 520 / **CPU 201** | **10.2 ms** | **16.5 ms** |
| 固定 H/W、batch 动态 | `batch×3×192×192` | CUDA 520 / **CPU 201** | 10.2 ms | 16.6 ms |
| 固定 H/W + 固定 batch=1 | `1×3×192×192` | CUDA 422 / **CPU 0** | **35.2 ms** | **51.2 ms** |

结论：

1. **只固定 H/W 完全没用**——CPU 节点数不变、耗时在噪声内（10.2 vs 10.2，16.5 vs 16.6）。
2. **形状算子是被动态 batch 拖住的**：一旦把 batch 固定成 1，CPU 节点归零（和
   admin 单独测到的 `CUDA 422 / CPU 0` 对得上），但**批处理也被一起废掉**：
   `OnnxBackend` 发现 `_batch_size == 1` 就逐张推理，6 个 ROI 变 3.4 倍慢、10 个 3.1 倍慢。
3. 所以**维持现状**：201 个形状算子的代价远小于丢掉批处理的代价。要真优化就固定
   H/W 的同时**保留动态 batch**——但那等于什么都不改（第 2 行）。

派生模型（不进 git，实验用，**未启用**）：

```zsh
.venv/bin/python tools/fix_onnx_shape.py \
    --model /home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/weights/armor/armor_best.onnx \
    --out   /home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/weights/armor/armor_best_fixed192.onnx \
    --size 192
# 源 sha256 463229640be82dc865f9bcf3199de9deb7a106831745dd3cd3a54c440b8de3de
# 派生 sha256 061534f0c77c8aba0b87d56c7e8693eef3d42677511700c99bbeb17033ca36b2
```

固定 batch 的对照模型用 `--fixed-batch 1` 生成（sha256 `a0062c54…`），只用于这次实验，
没有保留在权重目录里。

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
