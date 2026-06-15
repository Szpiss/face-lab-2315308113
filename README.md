# 人脸识别实验项目

本仓库整理人脸识别实验代码、数据说明、模型评估图表、Notebook 和答辩 PPT。由于仓库包含独立实验材料和私有课程信息，暂不并入公开深度学习仓库，保持为单独私有项目。

## 项目内容

| 文件/目录 | 说明 |
| --- | --- |
| `face_lab.py` | 人脸识别实验主程序 |
| `2315308113_徐颖莎_实验一.ipynb` | 实验 Notebook |
| `实验一_人脸识别系统_2315308113_徐颖莎.pptx` | 实验展示 PPT |
| `CASIA-FaceV5.zip` | 实验数据压缩包 |
| `artifacts/` | 指标表、混淆矩阵、ROC 曲线、样本图、聚类图等输出 |
| `requirements.txt` | Python 依赖 |
| `fill_ppt.py` | PPT 填充辅助脚本 |
| `update_notebook.py` | Notebook 更新辅助脚本 |

## 实验目标

项目围绕 CASIA 人脸数据集完成人脸识别流程，包含：

- 数据读取与样本预览
- 特征提取与降维分析
- SVM、随机森林、朴素贝叶斯等传统模型评估
- CNN 模型训练与结果比较
- DBSCAN 聚类可视化
- 混淆矩阵、ROC 曲线和指标表生成

## 运行方式

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python face_lab.py
```

如需重新生成或更新展示材料：

```bash
python update_notebook.py
python fill_ppt.py
```

## 输出说明

`artifacts/` 目录保存实验结果，包括：

- `metrics_summary.csv`、`metrics_summary.json`：模型指标汇总
- `metrics_bar.png`、`metrics_table.png`：指标可视化
- `*_cm.png`：混淆矩阵
- `*_roc.png`：ROC 曲线
- `pca_variance.png`：PCA 方差解释图
- `dbscan_kdist.png`、`dbscan_samples.png`：聚类分析图

## 维护说明

- 本仓库保留为独立私有实验项目，不与公开课程仓库合并。
- 若后续要公开，需要先确认数据、姓名、学号和课程材料是否可以公开。
- 新增图表或报告后，请同步更新本 README 的输出说明。
