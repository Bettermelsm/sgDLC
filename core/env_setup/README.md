# DeepLabCut 环境部署要点

面向在自有 GPU 服务器/工作站上部署 DLC 的通用步骤与选型依据。
以官方文档为准，本文记录的是我们实践中的关键决策点。

## 版本选择：2.3.x vs 3.0.x

- **生产环境建议 DLC 2.3.x**：行为稳定；模型权重不依赖 HuggingFace 在线拉取，
  对网络受限环境友好。
- **3.0.x**（截至本文写作仍处 rc 阶段，如 3.0.0rc9）存在若干已知 issue。若确需使用：
  1. 先过一遍上游仓库 issue 列表，确认你要用到的功能不在受影响清单里；
  2. 在自己的标注数据上**完整跑通 标注→训练→evaluate→analyze 链路**后再上生产。
- 两条在任何版本都成立的硬约束：
  - 跳过人工标注、直接用旧模型对新数据推理，结果没有科研效力；
  - 跳过 `evaluate_network` 直接推理，pcutoff 阈值就没有依据。

## GPU 环境搭建

1. `nvidia-smi` 确认驱动与 GPU 可见；
2. 按目标 torch 版本选 CUDA 构建（torch 自带 CUDA 运行时，驱动够新即可）；
3. 验证：
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

## 安装

conda / uv 均可，原则：给 DLC 一个独立环境，Python 取其支持区间内的较新版本。

```bash
# conda 方式（示例）
conda create -n dlc python=3.10 -y
conda activate dlc
pip install deeplabcut
```

装完后先 `import deeplabcut` 打印版本号，再进入自检清单。

## 网络受限环境（HuggingFace 下载阻塞）

- 设置镜像端点：`export HF_ENDPOINT=https://hf-mirror.com`；
- 或在能访问的机器上手动下载模型权重文件，拷贝到目标机对应缓存目录离线放置。

## 部署后自检清单

- [ ] `import deeplabcut` 成功且版本号符合预期
- [ ] 用小样例数据跑通 `create_training_dataset → train_network → evaluate_network → analyze_videos`
- [ ] `evaluate` 输出留档，作为下游 pcutoff 取值的依据
- [ ] 推理产物（h5/csv）的列结构下游能正确解析
- [ ] GPU 显存占用与批量推理并发数匹配（多路视频并行时留余量，避免 OOM）
