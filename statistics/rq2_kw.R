library(dplyr)
library(effsize)
library(FSA)

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
  baseline_alg <- baseline_static %>% filter(algorithm == alg)
  
  for (m in methods) {
    
    print(alg)
    print(m)
    
    m_data <- alg_data %>% filter(final_methods == m)
    wo_skip <- m_data %>% filter(ci_skip == 0)
    
    eval_data <- rbind(baseline_alg, wo_skip)

    res <- kruskal.test(builds_saved ~ final_methods, data = eval_data)
    print(res)
    res <- dunnTest(builds_saved ~ final_methods, data = eval_data , method = "bh")
    print(res)
  }
}