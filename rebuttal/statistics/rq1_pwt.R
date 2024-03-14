library(dplyr)
library(effsize)

build_data <- read.csv('/Users/divyakamath/Documents/Submitted Papers/EMSE - LWD/icsme/rebuttal/ci_skip_all_results.csv')

new_dynamic <- build_data %>% filter((method == "new_dynamic") & (ci_skip == 0))

new_dynamic['update_method_factor'] <- paste(new_dynamic$update_method, new_dynamic$factor)
new_dynamic['final_methods'] <- paste(new_dynamic$update_method_factor, new_dynamic$algorithm)

new_dynamic <- new_dynamic[new_dynamic$update_method != 'half_exp', ]

algorithms <- list("BATCHBISECT", "BATCHSTOP4", "BATCHDIVIDE4")
final_methods <- unique(new_dynamic$final_methods)
update_methods <- unique(new_dynamic$update_method)
update_method_factor <- unique(new_dynamic$update_method_factor)

random_methods <- list("random_linear", "random_exponential", "random_random")
non_random <- update_methods[!(update_methods %in% random_methods)]

# kruskal.test(builds_saved ~ final_methods, data = new_dynamic)
# pairwise.wilcox.test(new_dynamic$builds_saved, new_dynamic$final_methods,p.adjust.method = "BH")

for (alg in algorithms) {
  alg_data <- new_dynamic %>% filter(algorithm == alg)
  
  for (m in non_random) {
    print(alg)
    print(m)
    
    m_data <- alg_data %>% filter(update_method == m)
    
    res <- pairwise.wilcox.test(m_data$builds_saved, m_data$final_methods, p.adjust.method = "BH") 
    print(res)
    
    res <- cliff.delta(builds_saved ~ final_methods, data = m_data, return.dm=TRUE)
    print(res)
    
  }
  
  print(alg)
  print('random methods')
  random_data <- alg_data %>% filter(update_method %in% random_methods)
  
  res <- pairwise.wilcox.test(random_data$builds_saved, random_data$final_methods, p.adjust.method = "BH")
  print(res)
  
  res <- cliff.delta(builds_saved ~ final_methods, data = random_data, return.dm=TRUE)
  print(res)
  
}

bb_filters <- list('linear 1 BATCHBISECT', 'exponential 2 BATCHBISECT', 'stagger 2 BATCHBISECT', 'stagger_mfu 2 BATCHBISECT', 'random_random -1 BATCHBISECT')
bs4_filters <- list('linear 4 BATCHSTOP4', 'exponential 2 BATCHSTOP4', 'stagger 2 BATCHSTOP4', 'stagger_mfu 2 BATCHSTOP4', 'random_exponential -1 BATCHSTOP4')
bd4_filters <- list('linear 1 BATCHDIVIDE4', 'exponential 2 BATCHDIVIDE4', 'stagger 2 BATCHDIVIDE4', 'stagger_mfu 2 BATCHDIVIDE4', 'random_random -1 BATCHDIVIDE4')


# best_lwds <- list(bb_filters, bs4_filters, bd4_filters)
# for (lwd in best_lwds){
#   
#   eval_data <- new_dynamic %>% filter(final_methods %in% lwd)
#   print(length(eval_data$project))
#   
#   res <- kruskal.test(builds_saved ~ final_methods, data = eval_data)
#   print(res)
# 
#   res <- pairwise.wilcox.test(eval_data$builds_saved, eval_data$final_methods,p.adjust.method = "BH")
#   print(res)
#   
# }

