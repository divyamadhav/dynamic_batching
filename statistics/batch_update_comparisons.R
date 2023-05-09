build_data <- read.csv('batch_update_methods.csv')
baseline_data <- read.csv('baseline_dynamic_batching.csv')

library(dplyr)
group_by(build_data, update_method) %>%
  summarise(
    count = n(),
    mean = mean(builds_saved, na.rm = TRUE),
    sd = sd(builds_saved, na.rm = TRUE),
    median = median(builds_saved, na.rm = TRUE),
    IQR = IQR(builds_saved, na.rm = TRUE)
  )

algorithms <- list("BATCHBISECT", "BATCHSTOP4", "BATCHDIVIDE4")

methods <- list(build_data$update_method)
factors <- list(build_data$factor)
final_methods <- c(methods, factors)

build_data['final_methods'] <- final_methods

kruskal.test(builds_saved ~ final_methods, data = build_data)
pairwise.wilcox.test(build_data$builds_saved, build_data$final_methods,p.adjust.method = "BH")

for (alg in algorithms) {
  
  print(alg)
  print("--------")
  
  alg_data <- build_data %>% filter(algorithm == alg)
  baseline_alg <- baseline_data %>% filter(algorithm == alg)
  
  res <- kruskal.test(builds_saved ~ final_methods, data = alg_data)
  print(res)
  res <- pairwise.wilcox.test(alg_data$builds_saved, alg_data$final_methods,p.adjust.method = "BH")
  print(res)
  
  
  linear_2 <- alg_data %>% filter(update_method == "linear" & factor == "2")
  linear_3 <- alg_data %>% filter(update_method == "linear" & factor == "3")
  linear_4 <- alg_data %>% filter(update_method == "linear" & factor == "4")
  exponential_2 <- alg_data %>% filter(update_method == "exponential" & factor == "2") 
  exponential_3 <- alg_data %>% filter(update_method == "exponential" & factor == "3") 
  random_linear <- alg_data %>% filter(update_method == "random_linear" & factor == "-1") 
  random_exp <- alg_data %>% filter(update_method == "random_exponential" & factor == "-1") 
  stagger_2 <- alg_data %>% filter(update_method == "stagger" & factor == "2") 
  stagger_3 <- alg_data %>% filter(update_method == "stagger" & factor == "3") 
  
  res <- wilcox.test(linear_2$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(linear_3$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(linear_4$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(exponential_2$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(exponential_3$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(random_linear$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(random_exp$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(stagger_2$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
  res <- wilcox.test(stagger_3$builds_saved, baseline_alg$builds_saved, paired=TRUE)
  print(res)
  
}