library(dplyr)
library(effsize)


build_data <- read.csv('/Users/divyakamath/Documents/Submitted Papers/EMSE - LWD/icsme/rebuttal/ci_skip_all_results.csv')

static_batching <- build_data %>% filter(method == "baseline_static")
baseline_dynamic <- build_data %>% filter(method == "baseline_dynamic")
timeout_rule <- build_data %>% filter(method == "timeout_rule")
new_dynamic <- build_data %>% filter(method == "new_dynamic")

static_batching['final_methods'] <- paste(static_batching$algorithm, static_batching$batch_size)

algorithms <- list("BATCHBISECT", "BATCHSTOP4", "BATCHDIVIDE4")
final_methods <- unique(static_batching$final_methods)

for (m in final_methods) {
  
  s_alg <- static_batching %>% filter(final_methods == m)
  #d_alg <- baseline_dynamic %>% filter(final_methods == m)
  
  print(m)
  
  w_skip <- s_alg %>% filter(ci_skip == 1)
  wo_skip <- s_alg %>% filter(ci_skip == 0)
  
  N <- length(w_skip) + length(wo_skip)
  res <- wilcox.test(w_skip$builds_saved, wo_skip$builds_saved, p.adjust.method = "BH")
  print(res)
  res <- cliff.delta(w_skip$builds_saved, wo_skip$builds_saved,return.dm=TRUE)
  print(res)
  
}


for (alg in algorithms) {
  print(alg)
  d_alg <- baseline_dynamic %>% filter(algorithm == alg)
  w_skip <- d_alg %>% filter(ci_skip == 1)
  wo_skip <- d_alg %>% filter(ci_skip == 0)
  
  N <- length(w_skip) + length(wo_skip)
  res <- wilcox.test(w_skip$builds_saved, wo_skip$builds_saved, p.adjust.method = "BH")
  print(res)
  res <- cliff.delta(w_skip$builds_saved, wo_skip$builds_saved,return.dm=TRUE)
  print(res)
  
}