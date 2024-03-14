build_data <- read.csv('all_batchbisect.csv')

library(dplyr)
group_by(build_data, method) %>%
  summarise(
    count = n(),
    mean = mean(builds_saved, na.rm = TRUE),
    sd = sd(builds_saved, na.rm = TRUE),
    median = median(builds_saved, na.rm = TRUE),
    IQR = IQR(builds_saved, na.rm = TRUE)
  )

static_batching <- build_data %>% filter(method == "baseline_static" | method == "new_dynamic")
baseline_dynamic <- build_data %>% filter(method == "baseline_dynamic" | method == "new_dynamic")
new_dynamic <- build_data %>% filter(method == "new_dynamic")

kruskal.test(builds_saved ~ method, data = static_batching)
kruskal.test(builds_saved ~ method, data = baseline_dynamic)

N <- length()
res <- pairwise.wilcox.test(static_batching$builds_saved, static_batching$method,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra

N <- 94
res <- pairwise.wilcox.test(baseline_dynamic$builds_saved, baseline_dynamic$method,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra
