library(data.table)
library(arrow)

data <- as.data.table(read_parquet("./data/cleaned/claude_gemini_qwen_all.parquet"))
