library(dplyr)
library(effsize)
library(PMCMRplus)
library(DescTools)
library(FSA)
library(rstatix)

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

kruskal.test(builds_saved ~ final_methods, data = new_dynamic)
pairwise.wilcox.test(new_dynamic$builds_saved, new_dynamic$final_methods,p.adjust.method = "BH")

for (alg in algorithms) {
  alg_data <- new_dynamic %>% filter(algorithm == alg)

  for (m in non_random) {
    print(alg)
    print(m)

    m_data <- alg_data %>% filter(update_method == m)

    res <- friedman.test(builds_saved ~ final_methods | project, data = m_data)
    print(res)
    FT = xtabs(builds_saved ~ update_method_factor + project,
               data = m_data)

    K <- KendallW(FT, correct=TRUE, test=TRUE)
    print(K)
    CT = frdAllPairsConoverTest(y      = m_data$builds_saved,
                                groups = m_data$final_methods,
                                blocks = m_data$project,
                                p.adjust.method="BH")
    print(CT)

  }

  print(alg)
  print('random methods')
  random_data <- alg_data %>% filter(update_method %in% random_methods)

  res <- friedman.test(builds_saved ~ final_methods | project, data = random_data)
  print(res)

  FT = xtabs(builds_saved ~ update_method_factor + project,
             data = random_data)

  K <- KendallW(FT, correct=TRUE, test=TRUE)
  print(K)

  CT = frdAllPairsConoverTest(y      = random_data$builds_saved,
                              groups = random_data$final_methods,
                              blocks = random_data$project,
                              p.adjust.method="BH")
  print(CT)


}

bb_filters <- list('linear 1 BATCHBISECT', 'exponential 2 BATCHBISECT', 'stagger 2 BATCHBISECT', 'stagger_mfu 2 BATCHBISECT', 'random_random -1 BATCHBISECT')
bs4_filters <- list('linear 4 BATCHSTOP4', 'exponential 2 BATCHSTOP4', 'stagger 2 BATCHSTOP4', 'stagger_mfu 2 BATCHSTOP4', 'random_exponential -1 BATCHSTOP4')
bd4_filters <- list('linear 1 BATCHDIVIDE4', 'exponential 2 BATCHDIVIDE4', 'stagger 2 BATCHDIVIDE4', 'stagger_mfu 2 BATCHDIVIDE4', 'random_random -1 BATCHDIVIDE4')


best_lwds <- list(bb_filters, bs4_filters, bd4_filters)
for (lwd in best_lwds){
  
  print(lwd)

  eval_data <- new_dynamic %>% filter(final_methods %in% lwd)
  print(length(eval_data$project))
  
  res <- friedman.test(builds_saved ~ final_methods | project, data = eval_data)
  print(res)



  CT = frdAllPairsConoverTest(y      = eval_data$builds_saved,
                              groups = eval_data$final_methods,
                              blocks = eval_data$project,
                              p.adjust.method="BH")
  print(CT)

}

final_filter <- list('linear 1 BATCHBISECT', 'linear 4 BATCHSTOP4', 'exponential 2 BATCHDIVIDE4')
final_data <- new_dynamic %>% filter(final_methods %in% final_filter)
res <- friedman.test(builds_saved ~ final_methods | project, data = final_data)
print(res)
FT = xtabs(builds_saved ~ final_methods + project,
           data = final_data)

KendallW(FT, correct=TRUE, test=TRUE)
CT = frdAllPairsConoverTest(y      = final_data$builds_saved,
                            groups = final_data$final_methods,
                            blocks = final_data$project,
                            p.adjust.method="BH")
print(CT)



# for (alg in algorithms) {
#
#   alg_data <- new_dynamic %>% filter(algorithm == alg)
#
#   res <- friedman.test(builds_saved ~ update_method_factor | project, data = alg_data)
#   print(res)
#
#   FT = xtabs(builds_saved ~ update_method_factor + project,
#              data = alg_data)
#
#   K <- KendallW(FT, correct=TRUE, test=TRUE)
#   print(K)
#
#   CT = frdAllPairsConoverTest(y      = alg_data$builds_saved,
#                               groups = alg_data$update_method_factor,
#                               blocks = alg_data$project,
#                               p.adjust.method="BH")
#   print(CT)
# }