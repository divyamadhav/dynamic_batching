library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)
library(rstatix)


build_data <- read.csv('/Users/divyakamath/Documents/Submitted Papers/EMSE - LWD/icsme/rebuttal/ci_skip_all_results.csv')

time_data <- read.csv('/Users/divyakamath/Documents/Submitted Papers/EMSE - LWD/icsme/rebuttal/time_saved_all_methods.csv')
project_list <- unique(time_data$project)


build_data <- build_data %>% filter(project %in% project_list)

print(build_data$time_saved[2])
#build_data['total_duration'] <- 100 - build_data['time_saved']
print(build_data$time_saved[2])

new_dynamic <- build_data %>% filter((method == "new_dynamic") & (ci_skip == 0))
base_dynamic <- build_data %>% filter((method == "baseline_dynamic") & (ci_skip == 0))
base_static <- build_data %>% filter((method == "baseline_static") & (ci_skip == 0))


new_dynamic['update_method_factor'] <- paste(new_dynamic$update_method, new_dynamic$factor)
new_dynamic['final_methods'] <- paste(new_dynamic$update_method_factor, new_dynamic$algorithm)

base_static['final_methods'] <- paste(base_static$algorithm, base_static$batch_size)
base_static['update_method_factor'] <- base_static$batch_size

base_dynamic ['final_methods'] <- base_dynamic$algorithm
base_dynamic ['update_method_factor'] <- base_dynamic$algorithm

algorithms <- list("BATCHBISECT", "BATCHSTOP4", "BATCHDIVIDE4")
final_methods <- unique(new_dynamic$final_methods)
update_methods <- unique(new_dynamic$update_method)

random_methods <- list("random_linear", "random_exponential", "random_random")
non_random <- update_methods[!(update_methods %in% random_methods)]


for (i in 1:3) {

  alg <- algorithms[i]
  print(alg)

  base_alg <- base_static %>% filter(algorithm == alg)
  alg_data <- new_dynamic %>% filter(algorithm == alg)

  final_methods <- unique(alg_data$final_methods)

  for (j in 1:length(final_methods)){

    f <- final_methods[[j]]
    print(f)


    m_data <- alg_data %>% filter(final_methods == f)
    eval_data <- rbind(base_alg, m_data)


    res <- friedman.test(total_duration ~ final_methods | project, data = eval_data)
    print(res)

    FT = xtabs(total_duration ~ final_methods + project,
               data = eval_data)

    res = friedman_effsize(total_duration ~ final_methods | project, data = eval_data)
    print(res)

    CT = frdAllPairsConoverTest(y      = eval_data$total_duration,
                                groups = eval_data$final_methods,
                                blocks = eval_data$project,
                                p.adjust.method="bonferroni")
    print(CT)

  }

}


# for (i in 1:3) {
# 
#   alg <- algorithms[[i]]
# 
#   cat("Dynamic ", alg)
# 
#   base_alg <- base_dynamic %>% filter(algorithm == alg)
#   alg_data <- new_dynamic %>% filter(algorithm == alg)
# 
#   final_methods <- unique(alg_data$final_methods)
# 
#   for(j in 1:length(final_methods)){
# 
#     f <- final_methods[[j]]
#     print(f)
# 
# 
#     m_data <- alg_data %>% filter(final_methods == f)
# 
#     print(length(base_alg$total_duration))
#     print(length(m_data$total_duration))
# 
#     res <- wilcox.test(m_data$total_duration, base_alg$total_duration, paired=TRUE)
#     print(res)
#     res <- cliff.delta(m_data$total_duration, base_alg$total_duration,return.dm=TRUE)
#     print(res)
# 
# 
#   }
# 
# 
# }



# kruskal.test(builds_saved ~ final_methods, data = new_dynamic)
# pairwise.wilcox.test(new_dynamic$builds_saved, new_dynamic$final_methods,p.adjust.method = "BH")

# for (alg in algorithms) {
#   alg_data <- new_dynamic %>% filter(algorithm == alg)
#   
#   for (m in non_random) {
#     print(alg)
#     print(m)
#     m_data <- alg_data %>% filter(update_method == m)
#     res <- kruskal.test(builds_saved ~ final_methods, data = m_data)
#     print(res)
#     
#     res <- pairwise.wilcox.test(m_data$builds_saved, m_data$final_methods, p.adjust.method = "BH")
#     print(res)
#     
#   }
#   
#   print(alg)
#   print('random methods')
#   random_data <- alg_data %>% filter(update_method %in% random_methods)
#   res <- kruskal.test(builds_saved ~ final_methods, data = random_data)
#   print(res)
#   
#   res <- pairwise.wilcox.test(random_data$builds_saved, random_data$final_methods, p.adjust.method = "BH")
#   print(res)
#   
#   
# }

