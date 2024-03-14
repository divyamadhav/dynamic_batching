library(dplyr)
library(effsize)

build_data <- read.csv('/Users/divyakamath/Documents/Submitted Papers/EMSE - LWD/icsme/rebuttal/ci_skip_all_results.csv')

new_dynamic <- build_data %>% filter(method == "new_dynamic")
baseline_dynamic <- build_data %>% filter((method == "baseline_dynamic") & (ci_skip == 0))
baseline_static <- build_data %>% filter((method == "baseline_static") & (ci_skip == 0))

group_by(build_data, update_method) %>%
  summarise(
    count = n(),
    mean = mean(builds_saved, na.rm = TRUE),
    sd = sd(builds_saved, na.rm = TRUE),
    median = median(builds_saved, na.rm = TRUE),
    IQR = IQR(builds_saved, na.rm = TRUE)
  )

new_dynamic['update_method_factor'] <- paste(new_dynamic$update_method, new_dynamic$factor)
new_dynamic['final_methods'] <- paste(new_dynamic$update_method_factor, new_dynamic$algorithm)

algorithms <- list("BATCHBISECT", "BATCHDIVIDE4", "BATCHSTOP4")
methods <- unique(new_dynamic$final_methods)
# kruskal.test(builds_saved ~ final_methods, data = new_dynamic)
# pairwise.wilcox.test(new_dynamic$builds_saved, new_dynamic$final_methods,p.adjust.method = "BH")

bb_filters <- list('linear 1 BATCHBISECT', 'exponential 2 BATCHBISECT', 'stagger 2 BATCHBISECT', 'stagger_mfu 2 BATCHBISECT', 'random_random -1 BATCHBISECT')
bs4_filters <- list('linear 4 BATCHSTOP4', 'exponential 2 BATCHSTOP4', 'stagger 2 BATCHSTOP4', 'stagger_mfu 2 BATCHSTOP4', 'random_exponential -1 BATCHSTOP4')
bd4_filters <- list('linear 1 BATCHDIVIDE4', 'exponential 2 BATCHDIVIDE4', 'stagger 2 BATCHDIVIDE4', 'stagger_mfu 2 BATCHDIVIDE4', 'random_random -1 BATCHDIVIDE4')

best_lwds <- list(bb_filters, bd4_filters, bs4_filters)


for (i in 1:3) {
  
  alg <- algorithms[i]
  print('Beginning new algorithm')
  
  alg_data <- new_dynamic %>% filter(algorithm == alg)
  filters <- best_lwds[[i]]
  
  
  baseline_alg <- baseline_dynamic %>% filter(algorithm == alg)
  batch_sizes <- unique(baseline_alg$batch_size)
  
  for (m in filters) {
      cat("method", m)
      
      m_data <- alg_data %>% filter(final_methods == m)
      print(length(m_data$project))
      w_skip <- m_data %>% filter( ci_skip == 1)
      wo_skip <- m_data %>% filter( ci_skip == 0)
      print(length(baseline_alg$project))
      print(length(w_skip$project))
      print(length(wo_skip$project))
      
      # cat(m, "with ci_skip ", alg)
      # res <- wilcox.test(w_skip$builds_saved, baseline_alg$builds_saved, paired=TRUE)
      # print(res)
      # res <- cliff.delta(wo_skip$builds_saved, baseline_alg$builds_saved,return.dm=TRUE)
      # print(res)
      
      cat(m, "without ci_skip")
      res <- wilcox.test(wo_skip$builds_saved, baseline_alg$builds_saved, paired=TRUE)
      print(res)
      res <- cliff.delta(wo_skip$builds_saved, baseline_alg$builds_saved,return.dm=TRUE)
      print(res)

  }
}
