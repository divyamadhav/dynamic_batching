import random

def linear_update(res, batch, batch_size, factor, min_size=1, max_size=16):
    fails_per = 100*batch.count(0)/len(batch)
    upper_limit = max_size - factor
    if res == 0:
        
        if fails_per > 40:
            return min_size
        
        if (fails_per < 20) & (batch[-1] == 1):
            return batch_size

        #if batch has more than 20% failures then definitely split
        if batch_size <= factor:
            return min_size
        else:
            batch_size -= factor


    else:
        
        if batch_size > upper_limit:
            return batch_size
        else:
            batch_size += factor
    
    return batch_size



def exp_update(res, batch, batch_size, factor, min_size=1, max_size=16):
    fails_per = 100*batch.count(0)/len(batch)
    upper_limit = max_size // factor
    if res == 0:
        
        if fails_per > 40:
            return min_size

        if (fails_per < 20) & (batch[-1] == 1):
            return batch_size
        
        #if batch has more than 20% failures then definitely split
        if batch_size <= factor:
            return min_size
        else:
            batch_size //= factor


    else:
        
        if batch_size > upper_limit:
            return batch_size
        else:
            batch_size *= factor
    
    return batch_size


def half_exp_update(res, batch, batch_size, factor, min_size=1, max_size=16):
    fails_per = 100*batch.count(0)/len(batch)
    upper_limit = max_size
    
    if res == 0:
        
        if fails_per > 40:
            return min_size
        
        if (fails_per < 20) & (batch[-1] == 1):
            return batch_size
        
        batch_size -= int(batch_size/factor)
    else:
        
        if batch_size > upper_limit:
            return batch_size
        else:
            batch_size += int(batch_size/factor)
            
    return batch_size

def stagger_update(res, batch, batch_size, factor, min_size=1):
    
    fails_per = 100*batch.count(0)/len(batch)
    
    if res == 0:
        
        if fails_per > 40:
            return min_size
        
        if batch_size <= factor:
            return min_size
        
        if (fails_per < 10) & (batch[-1] == 1):
            return batch_size
        if (fails_per < 20) & (batch[-1] == 1):
            return batch_size - factor
        if (fails_per < 50) & (batch[-1] == 1):
            return batch_size // factor
        else:
            batch_size = 1


    else:
        
        if batch_size >= 16:
            return 16
        else:
            batch_size = min(batch_size * factor, 16)
    
    return batch_size



def random_linear(res, batch, batch_size, factor, min_size=1, max_size=16):
    fails_per = 100*batch.count(0)/len(batch)
    
    
    
    if res == 0:
        
        if fails_per > 40:
            return min_size
                
        factor = random.randint(1, batch_size)

        if (fails_per < 20) & (batch[-1] == 1):
            return batch_size
        
        #if batch has more than 20% failures then definitely split
        if batch_size <= factor:
            return min_size
        else:
            batch_size -= factor


    else:
        
        if batch_size >= 16:
            return batch_size
        else:
            factor = random.randint(batch_size, max_size)
            print('Factor = {}'.format(str(factor)))
            batch_size += factor
    
    return batch_size



def random_exp(res, batch, batch_size, factor, min_size=1):
    fails_per = 100*batch.count(0)/len(batch)
    
    if res == 0:   
        
        if fails_per > 40:
            return min_size
        
        
        factor = random.randint(1, batch_size)
        
        if (fails_per < 20) & (batch[-1] == 1):
            return batch_size
        
        #if batch has more than 20% failures then definitely split
        if batch_size <= factor:
            return min_size
        else:
            batch_size //= factor


    else:
        
        factor = random.randint(2, 4)
        
        if batch_size >= 16:
            return batch_size
        else:
            batch_size *= factor
    
    return batch_size


def random_random(res, batch, batch_size, factor, min_size=1, max_size=16):
    fails_per = 100*batch.count(0)/len(batch)
    
    if res == 0:
        
        if fails_per > 40:
            return min_size
        
        if (fails_per < 20) & (batch[-1] == 1):
            return batch_size
        
        batch_size = random.randint(min_size, batch_size)
        
    else:
        
        batch_size = random.randint(batch_size, max_size)
        
    return batch_size



def mfu(results, factor, algorithm):
    
    max_batch_size = 16
    min_batch_size = 1
    batch_sizes = []
    
    i = 0
    builds = 0
    length = len(results)
    
    cur_batch_size = max_batch_size
    
    while i < length:
        
        batch = results[i:i+cur_batch_size]
        i = i+cur_batch_size
        batch_sizes.append(cur_batch_size)
        
        batch_total = algorithm(batch)
        builds += batch_total
        
        if 0 in batch:
            res = 0
        else:
            res = 1
        
        fails_per = 100*batch.count(0)/len(batch)
        #print(cur_batch_size, batch, batch_total, fails_per)
        
        if res == 0:
        
            if cur_batch_size <= factor:
                cur_batch_size = min_batch_size

            elif fails_per < 20:
                cur_batch_size -= factor

            elif fails_per < 50:
                cur_batch_size //= factor

            else:
                cur_batch_size = Counter(batch_sizes).most_common(1)[0][0]
        
        else:
            
            if cur_batch_size >= 16:
                cur_batch_size = max_batch_size
            else:
                cur_batch_size = min(cur_batch_size * factor, 16)
        
    
    builds_saved = 100-(100*builds/length)
    return builds, builds_saved