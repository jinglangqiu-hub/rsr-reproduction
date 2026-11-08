# 原始来源与修改说明

本项目基于 Fuli Feng 等作者的 Temporal Relational Ranking for Stock Prediction / Relational Stock Ranking 实现。

- 来源：https://github.com/fulifeng/Temporal_Relational_Stock_Ranking
- 固定提交：`cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d`
- 上游许可证：GNU Affero General Public License, Version 3，原文复制至本项目 `LICENSE`。
- 原始实现通过 `upstream/` 子模块保留；`compat/training/` 为该提交的适配副本，原始文件散列记录于 `audit/source-manifest.json`。

兼容修改及独立实验工作于 2026 年 11 月完成，主要包括：

1. 将旧版 TensorFlow 接口迁移到 TensorFlow 2.21 的 compat.v1 图模式及内部旧层实现，保留原模型结构与损失。
2. 关系矩阵采用内存映射和非训练 GPU 变量，减少图序列化及内存开销；保留 FP32 对照核验。
3. 新增实验配置冻结、GPU 启动器、断点恢复、浏览器监测、数据核验与绘图。
4. 原作者流程复现之后，另行执行限制训练标签边界、训练期最高价归一化的超参数搜索及五种子实验；这些变更在项目文档中单独说明。

原作者 README、代码、许可证和论文引用的权利及归属保持不变。引用原模型时请使用 `CITATION.bib`，并说明使用了本复现项目的兼容实现及具体实验协议。
