# ECFNet 项目结构与执行流程

`config/index.yaml` 是模型结构和基础参数的统一来源。`config/index-strict.yaml` 继承基础配置，启用独立验证被试与训练集拟合的归一化；`config/ablations/` 继承严格配置。

## 模块职责

- `main.py`：配置加载、日志、随机种子、预处理、训练或检查点评估入口。
- `cross_validation.py`：按被试划分训练/验证/测试，归一化、训练、恢复最佳模型、导出每折结果与汇总。
- `datapipe.py`：读取 SEED MATLAB 试次、数值排序、特征缓存、按被试加载。
- `utils/transform.py`：通道重排、试次分段和子窗口切片。
- `utils/feature_process.py`：DE 等低阶特征；DE 使用总体方差。
- `utils/wavelet.py`：明确区分 a4/d4/d3/d2/d1，按论文式 (2)–(3) 计算三个固定描述符。
- `utils/channel.py`：62 电极顺序与 11/14/19 区域划分；original 为 62 个单节点区域。
- `utils/config.py`：配置继承、参数结构与检查。
- `net/model.py`：ECFNet、局部映射、动态图、区域与频带池化、定向注意力；保留旧模型入口。
- `net/layer.py`：SGC、多头注意力、FFN、查询向量池化。
- `net/util.py`：模型工厂，所有模型通过工厂注册创建。
- `train.py`：逐轮训练、评估、早停、检查点与 TensorBoard。
- `visualize.py`：可选解释工具；使用当前留出被试和已保存的归一化参数。
- `tests/`：公式、结构、数据管线、评估协议和检查点复现检查。
- `scripts/make_release.py`：生成排除数据、日志和本地缓存的 GitHub 上传包。

## 执行路径

`main.py → cross_validation.py → datapipe.py / train.py / net/`

首次运行构建 `data/processed/SEED/<config_name>/{data.h5,index.parquet,config.json}`。后续根据预处理指纹决定是否重建。改变原始数据内容但保持原路径时，应显式使用 `--rebuild-data`。

严格模式每折使用 13 名训练被试、1 名验证被试、1 名测试被试。验证被试固定为测试被试的下一个编号，15 的下一个为 1。归一化仅拟合训练被试，验证指标用于选最佳轮次；恢复检查点后才计算测试指标。历史模式使用 14 名训练被试与测试集选轮次，并明确标记为 `legacy_test`。

`--folds` 只限制运行哪些测试折，不缩减训练被试池。每折随机种子为 `random_seed + subject_id`。完整配置保存在 records；重复覆盖已完成折、同实验名混入另一配置均会被拒绝。

## 输出位置

- 特征缓存：`data/processed/SEED/<config_name>/`。
- 检查点：`run/checkpoints/<exp_name>/`。
- 配置、划分、归一化参数、预测、指标、图：`run/records/<exp_name>/`。
- TensorBoard：`run/tensorboard/<exp_name>/loso_sub_<id>/`。
- 日志：`run/logs/<save_dir>/`。
- 重新评估：`run/records/<exp_name>/evaluation/`。

为新实验设置独立的 `exp_name` 与日志 `save_dir`；改变缓存结构时设置独立的 `config_name`。复评必须同时保留检查点和 records，并使用训练时相同的配置。
