library(dplyr)

build_data <- read.csv('/Users/divyakamath/Documents/icsme/dynamic batching/ci_skip_final_results.csv')

static_batching <- build_data %>% filter(method == "baseline_static")
baseline_dynamic <- build_data %>% filter(method == "baseline_dynamic")
timeout_rule <- build_data %>% filter(method == "timeout_rule")
new_dynamic <- build_data %>% filter(method == "new_dynamic")

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
final_methods <- unique(new_dynamic$final_methods)

for (alg in algorithms) {
  
  s_alg <- static_batching %>% filter(algorithm == alg)
  t_alg <- timeout_rule %>% filter(algorithm == alg)
  d_alg <- baseline_dynamic %>% filter(algorithm == alg)
  n_alg <- new_dynamic %>% filter(algorithm == alg)
  
  for (m in final_methods){
    
    m_data <- n_alg %>% filter(final_methods == m & ci_skip == 0)
    m_data <- subset(m_data, select=-c(final_methods))
    
    batch_sizes <- unique(s_alg$batch_sizes)
    for (b in batch_sizes){
      s_b <- s_alg %>% filter(batch_size == b)
      
      df <- rbind(m_data, s_b)
      N <- length(df$project)
      
      kruskal.test(builds_saved ~ method, data = df)
      
      res <- pairwise.wilcox.test(df$builds_saved, df$method,p.adjust.method = "BH")
      print(res)
      Za = qnorm(res$p.value/2)
      ra = abs(Za)/sqrt(N)
      ra
    }
    
    batch_sizes <- unique(t_alg$batch_sizes)
    for (b in batch_sizes){
      t_b <- t_alg %>% filter(batch_size == b)
      df <- rbind(m_data, t_b)
      N <- length(df$project)
      
      kruskal.test(builds_saved ~ method, data = df)
      
      res <- pairwise.wilcox.test(df$builds_saved, df$method,p.adjust.method = "BH")
      res
      Za = qnorm(res$p.value/2)
      ra = abs(Za)/sqrt(N)
      ra
    }
    
    df <- rbind(m_data, d_alg)
    kruskal.test(builds_saved ~ method, data = df)
    
    res <- pairwise.wilcox.test(df$builds_saved, df$method,p.adjust.method = "BH")
    res
    Za = qnorm(res$p.value/2)
    ra = abs(Za)/sqrt(N)
    ra
    
  }
  
}
