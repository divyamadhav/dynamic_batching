library(dplyr)
library(effsize)
build_data <- read.csv('/Users/divyakamath/Documents/icsme/dynamic batching/ci_skip_final_results.csv')

new_dynamic <- build_data %>% filter(method == "new_dynamic")
baseline_dynamic <- build_data %>% filter(method == "baseline_dynamic")
baseline_static <- build_data %>% filter(method == "baseline_static")

group_by(build_data, update_method) %>%
  summarise(
    count = n(),
    mean = mean(builds_saved, na.rm = TRUE),
    sd = sd(builds_saved, na.rm = TRUE),
    median = median(builds_saved, na.rm = TRUE),
    IQR = IQR(builds_saved, na.rm = TRUE)
  )

new_dynamic['final_methods'] <- paste(new_dynamic$update_method, new_dynamic$factor)

algorithms <- list("BATCHBISECT", "BATCHSTOP4", "BATCHDIVIDE4")
methods <- unique(new_dynamic$final_methods)

new_dynamic <- subset(new_dynamic, update_method != 'half_exp')

for (alg in algorithms) {
  
  alg_data <- new_dynamic %>% filter(algorithm == alg)
  
  print(alg)
  res <- kruskal.test(builds_saved ~ final_methods, data = alg_data)
  print(res)
  
  res <- pairwise.wilcox.test(alg_data$builds_saved, alg_data$final_methods,p.adjust.method = "BH")
  print(res)
  N <- length(alg_data$builds_saved)
  Za = qnorm(res$p.value/2)
  ra = abs(Za)/sqrt(N)
  print(ra)
  
  res <- cliff.delta(builds_saved ~ final_methods, data = alg_data, return.dm=TRUE)
  print(res)
  
  # for (m in methods) {
  #   
  #   print(alg)
  #   print(m)
  #   
  #   m_data <- alg_data %>% filter(final_methods == m)
  #   w_skip = m_data %>% filter(ci_skip == 1)
  #   wo_skip = m_data %>% filter(ci_skip == 0)
  #   
  #   res <- wilcox.test(w_skip$builds_saved, wo_skip$builds_saved, paired=TRUE)
  #   print(res)
  #   # N <- length(w_skip$project) + length(wo_skip$project)
  #   # Za = qnorm(res$p.value/2)
  #   # ra = abs(Za)/sqrt(N)
  #   # print(ra)
  #   res <- cliff.delta(w_skip$builds_saved, wo_skip$builds_saved,return.dm=TRUE)
  #   print(res)
  #   
  # }
}

