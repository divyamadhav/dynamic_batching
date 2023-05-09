build_data <- read.csv('ci_skip_final_results.csv')

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
timeout_rule <- build_data %>% filter(method == "timeout_rule" | method == "new_dynamic")
new_dynamic <- build_data %>% filter(method == "new_dynamic")

kruskal.test(builds_saved ~ method, data = static_batching)
kruskal.test(builds_saved ~ method, data = baseline_dynamic)

N <- 94
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

timeout_projects <- unique(build_data[build_data$method == "timeout_rule", "project"])
timeout_new_dynamic <- filter(new_dynamic, project %in% timeout_projects)

temp_projects <- unique(timeout_new_dynamic[, "project"])
new_timeout_rule <- filter(timeout_rule, project %in% temp_projects)

kruskal.test(builds_saved ~ method, data = new_timeout_rule)
N <- 36
res <- pairwise.wilcox.test(new_timeout_rule$builds_saved, new_timeout_rule$method,p.adjust.method = "BH")
res
Za = qnorm(res$p.value/2)
ra = abs(Za)/sqrt(N)
ra

# baseline <- ssr_batch4 %>% filter(Group == "Batch4_Builds" )
# hybrid <- ssr_batch4 %>% filter(Group == "SSR_Batch4_Builds" )
# res <- wilcox.test(baseline$BuildsRequired, hybrid$BuildsRequired, p.adjust.method = "BH")
# res
# Za = qnorm(res$p.value/2)
# ra = abs(Za)/sqrt(N)
# ra
