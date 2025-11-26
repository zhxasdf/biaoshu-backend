## 项目简介

本仓库包含一个面向招投标文档的后端生成流程：解析原始标书 → 结构化提取 → 调用大模型扩写与润色 → 输出 Markdown/Word 成品。入口脚本为 `run_main.py`，主要依赖 `DocxParser`（解析阶段）与 `FileGenerate`（生成阶段）。

## 环境准备
- Python 3.10+（建议使用虚拟环境）


```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## 模型接口配置
1. 复制示例配置：
   ```bash
   cp configs/llm_apis.example.json configs/llm_apis.json
   ```
2. 按需填写各模型的 `base_url` 与 `api_key`。

示例结构：
```json
{
  "qwen-plus": {
    "base_url": "https://api.example.com/v1",
    "api_key": "YOUR_API_KEY"
  }
}
```

## 运行步骤
2. 执行：
   ```bash
   python run_main.py -c configs/config_v1.yaml
   ```
3. 运行日志、阶段性 JSON/Markdown/Word 产物会写入 `outputs/<原始文件名>/`。

## 输出内容
- `step4_score_table.json`：评分表结构化数据
- `step5_st_tech_spec.json`：评分表 + 技术建议书 + 技术规范融合结果
- `step6_expand_dir.*`：扩展后的目录
- `step6_final_res.*`：填充完成的正文
- `step6_polished_res.*`：二次润色后的终稿



按以上流程配置后，拉取仓库的新同事只需准备虚拟环境、补齐模型密钥就能直接运行整个后端管线。***

