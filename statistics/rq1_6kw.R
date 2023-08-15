library(dplyr)
library(effsize)

build_data <- read.csv('/Users/divyakamath/Documents/icsme/dynamic batching/ci_skip_all_results.csv')

new_dynamic <- build_data %>% filter((method == "new_dynamic") & (ci_skip == 0))

new_dynamic['update_method_factors'] <- paste(new_dynamic$update_method, new_dynamic$factor)
new_dynamic['final_methods'] <- paste(new_dynamic$update_method_factors, new_dynamic$algorithm)

new_dynamic <- new_dynamic[new_dynamic$update_method != 'half_exp', ]

algorithms <- list("BATCHBISECT", "BATCHSTOP4", "BATCHDIVIDE4")
final_methods <- unique(new_dynamic$final_methods)

update_methods <- unique(new_dynamic$update_method)

random_methods <- list("random_linear", "random_exponential", "random_random")
non_random <- update_methods[!(update_methods %in% random_methods)]
print(non_random)



for (m in non_random) {
  print(m)
  method_data <- new_dynamic %>% filter(update_method == m)

  res <- kruskal.test(builds_saved ~ final_methods, data = method_data)
  print(res)

  res <- pairwise.wilcox.test(method_data$builds_saved, method_data$final_methods,p.adjust.method = "BH")
  print(res)
}

print('random methods')
random_data <- new_dynamic %>% filter(update_method %in% random_methods)
res <- kruskal.test(builds_saved ~ final_methods, data = random_data)
print(res)

res <- pairwise.wilcox.test(random_data$builds_saved, random_data$final_methods,p.adjust.method = "BH")
print(res)


# bb_filters <- list('linear 2 BATCHBISECT', 'exponential 3 BATCHBISECT', 'stagger 3 BATCHBISECT', 'stagger_mfu 3 BATCHBISECT', 'random_random -1 BATCHBISECT')
# bs4_filters <- list('linear 4 BATCHSTOP4', 'exponential 3 BATCHSTOP4', 'stagger 3 BATCHSTOP4', 'stagger_mfu 3 BATCHSTOP4', 'random_linear -1 BATCHSTOP4')
# bd4_filters <- list('linear 1 BATCHDIVIDE4', 'exponential 3 BATCHDIVIDE4', 'stagger 3 BATCHDIVIDE4', 'stagger_mfu 3 BATCHDIVIDE4', 'random_exponential -1 BATCHDIVIDE4')
# 
# 
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