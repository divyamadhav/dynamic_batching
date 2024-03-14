library(dplyr)
library(effsize)
library(FSA)
library(PMCMRplus)

build_data <- read.csv('/Users/divyakamath/Documents/Submitted Papers/EMSE - LWD/icsme/rebuttal/ci_skip_all_results.csv')

new_dynamic <- build_data %>% filter((method == "new_dynamic") & (ci_skip == 0))
base_static <- build_data %>% filter((method == "baseline_static") & (ci_skip == 0))

new_dynamic <- new_dynamic[new_dynamic$update_method != 'half_exp', ]

new_dynamic['update_method_factor'] <- paste(new_dynamic$update_method, new_dynamic$factor)
new_dynamic['final_methods'] <- paste(new_dynamic$update_method_factor, new_dynamic$algorithm)

base_static['final_methods'] <- paste(base_static$algorithm, base_static$batch_size)
base_static['update_method_factor'] <- base_static$batch_size

algorithms <- list("BATCHBISECT", "BATCHSTOP4", "BATCHDIVIDE4")
final_methods <- unique(new_dynamic$final_methods)
update_methods <- unique(new_dynamic$update_method)
update_method_factor <- unique(new_dynamic$update_method_factor)

bb_filters <- list('linear 1 BATCHBISECT', 'exponential 2 BATCHBISECT', 'stagger 2 BATCHBISECT', 'stagger_mfu 2 BATCHBISECT', 'random_random -1 BATCHBISECT')
bs4_filters <- list('linear 4 BATCHSTOP4', 'exponential 2 BATCHSTOP4', 'stagger 2 BATCHSTOP4', 'stagger_mfu 2 BATCHSTOP4', 'random_exponential -1 BATCHSTOP4')
bd4_filters <- list('linear 1 BATCHDIVIDE4', 'exponential 2 BATCHDIVIDE4', 'stagger 2 BATCHDIVIDE4', 'stagger_mfu 2 BATCHDIVIDE4', 'random_random -1 BATCHDIVIDE4')

best_lwds <- list(bb_filters, bs4_filters, bd4_filters)

pdata = matrix(ncol=11, nrow=7)


random_methods <- list("random_linear", "random_exponential", "random_random")
non_random <- update_methods[!(update_methods %in% random_methods)]

list_range <- 1:3

for (i in list_range) {
  
  alg <- algorithms[i]
  f <- best_lwds[[i]]
  
  alg_data <- new_dynamic %>% filter(algorithm == alg)
  m_data <- alg_data %>% filter(final_methods %in% f)
  base_alg <- base_static %>% filter(algorithm == alg)


  print(alg)

  eval_data <- rbind(base_alg, m_data)

  res <- friedman.test(builds_saved ~ final_methods | project, data = eval_data)
  print(res)
  
  FT = xtabs(builds_saved ~ final_methods + project,
             data = eval_data)
  
  res = KendallW(FT, correct=TRUE, test=TRUE)
  print(res)
  CT = frdAllPairsConoverTest(y      = eval_data$builds_saved,
                              groups = eval_data$final_methods,
                              blocks = eval_data$project,
                              p.adjust.method="bonferroni")
  print(CT)
}


#Comparing with Random Methods

bb_r = 'random_random -1 BATCHBISECT'
bs4_r = 'random_linear -1 BATCHSTOP4'
bd4_r = 'random_exponential -1 BATCHDIVIDE4'

best_randoms <- list(bb_r, bs4_r, bd4_r)


list_range <- 1:3

for (i in list_range) {
  
  alg <- algorithms[i]
  f <- best_randoms[[i]]
  
  alg_data <- new_dynamic %>% filter(algorithm == alg)
  m_data <- alg_data %>% filter(final_methods %in% f)
  
  base_alg <- base_static %>% filter(algorithm == alg)
  
  
  print(alg)
  
  eval_data <- rbind(base_alg, m_data)
  
  res <- friedman.test(builds_saved ~ final_methods | project, data = eval_data)
  print(res)
  
  FT = xtabs(builds_saved ~ final_methods + project,
             data = eval_data)
  
  res = KendallW(FT, correct=TRUE, test=TRUE)
  print(res)
  CT = frdAllPairsConoverTest(y      = eval_data$builds_saved,
                              groups = eval_data$final_methods,
                              blocks = eval_data$project,
                              p.adjust.method="bonferroni")
  print(CT)
}


#Comparing with Random Methods

bb_r = 'random_random -1 BATCHBISECT'
bs4_r = 'random_linear -1 BATCHSTOP4'
bd4_r = 'random_exponential -1 BATCHDIVIDE4'

best_randoms <- list(bb_r, bs4_r, bd4_r)


list_range <- 1:3

for (i in list_range) {
  
  alg <- algorithms[i]
  f <- best_randoms[[i]]
  
  alg_data <- new_dynamic %>% filter(algorithm == alg)
  m_data <- alg_data %>% filter(final_methods %in% f)
  
  base_alg <- base_static %>% filter(algorithm == alg)
  
  
  print(alg)
  
  eval_data <- rbind(base_alg, m_data)
  
  res <- friedman.test(builds_saved ~ final_methods | project, data = eval_data)
  print(res)
  
  FT = xtabs(builds_saved ~ final_methods + project,
             data = eval_data)
  
  res = KendallW(FT, correct=TRUE, test=TRUE)
  print(res)
  CT = frdAllPairsConoverTest(y      = eval_data$builds_saved,
                              groups = eval_data$final_methods,
                              blocks = eval_data$project,
                              p.adjust.method="bonferroni")
  print(CT)
}