library(dplyr)
library(effsize)

build_data <- read.csv('/Users/divyakamath/Documents/icsme/dynamic batching/ci_skip_final_results.csv')

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

new_dynamic['final_methods'] <- paste(new_dynamic$update_method, new_dynamic$factor)

algorithms <- list("BATCHBISECT", "BATCHDIVIDE4", "BATCHSTOP4")
methods <- unique(new_dynamic$final_methods)
methods <- list("random_random -1")
kruskal.test(builds_saved ~ final_methods, data = new_dynamic)
pairwise.wilcox.test(new_dynamic$builds_saved, new_dynamic$final_methods,p.adjust.method = "BH")

for (alg in algorithms) {

  alg_data <- new_dynamic %>% filter(algorithm == alg)
  
  # res <- kruskal.test(builds_saved ~ final_methods, data = alg_data)
  # print(res)
  # res <- pairwise.wilcox.test(alg_data$builds_saved, alg_data$final_methods,p.adjust.method = "BH")
  # print(res)
  
  baseline_alg <- baseline_dynamic %>% filter(algorithm == alg)
  batch_sizes <- unique(baseline_alg$batch_size)
  
  for (m in methods) {
      
      m_data <- alg_data %>% filter( final_methods == m)
      
      w_skip <- m_data %>% filter( ci_skip == 1)
      wo_skip <- m_data %>% filter( ci_skip == 0)
      print(length(baseline_alg$project))
      print(length(w_skip$project))
      print(length(wo_skip$project))
      
      # cat(alg, ": batch_size =" , b, "\n")
      # cat(m, "with ci_skip")
      # res <- wilcox.test(w_skip$builds_saved, baseline_alg$builds_saved, paired=TRUE)
      # print(res)
      # res <- cliff.delta(wo_skip$builds_saved, baseline_alg$builds_saved,return.dm=TRUE)
      # print(res)
      
      #cat(alg, ": batch_size =" , b, "\n")
      cat(m, "without ci_skip")
      res <- wilcox.test(wo_skip$builds_saved, baseline_alg$builds_saved, paired=TRUE)
      print(res)
      # res <- cliff.delta(wo_skip$builds_saved, baseline_alg$builds_saved,return.dm=TRUE)
      # print(res)

  }
}
