# Run from the 448Midterm project in RStudio after arrow finishes installing.
library(arrow)
library(data.table)

dataset_path <- "data/cleaned/claude_gemini_qwen_balanced.parquet"
if (!file.exists(dataset_path)) {
  stop("Open 448Midterm.Rproj and run this script from the project directory.")
}

responses <- as.data.table(read_parquet(dataset_path))
stopifnot(nrow(responses) == 9000L, !anyDuplicated(responses$example_id))

print(responses[, .(
  responses = .N,
  median_prompt_words = median(prompt_word_count),
  median_response_words = median(response_word_count),
  mean_response_words = round(mean(response_word_count), 1)
), by = .(LLM_name, model_version)])

boxplot(response_word_count ~ LLM_name, data = responses,
        outline = FALSE, col = c("#226F91", "#C65D3B", "#5D8B52"),
        main = "Response lengths by LLM family", ylab = "Words", xlab = "LLM family")

split_path <- "results/split_manifest.parquet"
if (file.exists(split_path)) {
  assignments <- as.data.table(read_parquet(split_path))
  print(assignments[, .N, by = .(split, LLM_name)][order(split, LLM_name)])
  stopifnot(assignments[, uniqueN(split), by = split_group][, all(V1 == 1L)])
}

# Use LLM_Input and/or LLM_output as model inputs. LLM_name is the target.
# Keep all examples with the same split_group in the same split.
# The Python model implementations are in scripts/train_experiments.py.
