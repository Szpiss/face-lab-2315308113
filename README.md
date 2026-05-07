# 实验一 人脸识别系统

学号：2315308113  
姓名：徐颖莎

## 项目内容

本实验基于 CASIA-FaceV5 数据集完成人脸识别任务，主要流程包括：

1. 读取并展示人脸图像
2. 使用 PCA 进行降维
3. 使用 DBSCAN 去除噪声样本
4. 训练朴素贝叶斯、支持向量机、随机森林、卷积神经网络四种模型
5. 对比四种模型的准确率、查准率、召回率、F1 和 AUC
6. 生成实验图片并填写实验报告 Notebook 和 PPT

## 目录说明

- `2315308113_徐颖莎_实验一.ipynb`：完成版实验 Notebook
- `实验一_人脸识别系统_2315308113_徐颖莎.pptx`：填写后的实验 PPT
- `face_lab.py`：实验主要函数
- `fill_ppt.py`：PPT 自动填充脚本
- `update_notebook.py`：Notebook 自动填写脚本
- `artifacts/`：实验中生成的图片和指标结果
- `CASIA-FaceV5.zip`：原始数据压缩包

## 运行方式

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
unzip -o CASIA-FaceV5.zip
.venv/bin/python update_notebook.py
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace 2315308113_徐颖莎_实验一.ipynb
.venv/bin/python fill_ppt.py
```

## 结果摘要

- PCA 最终保留 278 维，累计解释方差约 95%
- DBSCAN 参数为 `eps=24`、`min_samples=5`
- 共识别出 28 个噪声样本
- 四种模型中，SVM 的准确率最高，约为 0.3571

