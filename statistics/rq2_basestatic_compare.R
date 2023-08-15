library(dplyr)
library(effsize)
build_data <- read.csv('/Users/divyakamath/Documents/icsme/dynamic batching/ci_skip_all_results.csv')

new_dynamic <- build_data %>% filter((method == "new_dynamic"))
baseline_dynamic <- build_data %>% filter((method == "baseline_dynamic") & (ci_skip == 0))
baseline_static <- build_data %>% filter((method == "baseline_static") & (ci_skip == 0))

new_dynamic <- new_dynamic[new_dynamic$update_method != 'half_exp', ]
new_dynamic['final_methods'] <- paste(new_dynamic$update_method, new_dynamic$factor)
baseline_static['final_methods'] <- paste(baseline_static$algorithm, baseline_static$batch_size)

algorithms <- list("BATCHBISECT", "BATCHDIVIDE4", "BATCHSTOP4")
methods <- unique(new_dynamic$final_methods)

for (alg in algorithms) {

  alg_data <- new_dynamic %>% filter(algorithm == alg)

  print(alg)
  baseline_alg <- baseline_static %>% filter(algorithm == alg)
  batch_sizes <- unique(baseline_alg$batch_size)

  for (b in batch_sizes) {

    b_data <- baseline_alg %>% filter(batch_size == b)
    print(length(b_data$project))
    for (m in methods) {

      m_data <- alg_data %>% filter( final_methods == m)

      w_skip <- m_data %>% filter(ci_skip == 1)
      wo_skip <- m_data %>% filter(ci_skip == 0)
      print(length(w_skip$project))
      print(length(wo_skip$project))

      cat(alg, ": batch_size =" , b, "\n")
      cat(m, "without ci_skip")
      res <- wilcox.test(wo_skip$builds_saved, b_data$builds_saved, paired=TRUE, p.adjust.method ="bonferroni")
      print(res)
      # res <- cliff.delta(wo_skip$builds_saved, b_data$builds_saved,return.dm=TRUE)
      # print(res)
    }
  }
  break
}

