batch_duration = 0

def batchstop4(commits):
    
    l = len(commits)//2
    
    if 0 in commits:
        
        if len(commits) < 4:
            return len(commits)
        
        if len(commits) == 4:
            return 1 + len(commits)
        
        return 1 + batchstop4(commits[:l]) + batchstop4(commits[l:])
    
    return 1

def batchbisect(commits):
    
    l = len(commits)//2
    
    
    if len(commits) == 1:
        return 1
    
    if 0 in commits:
        return 1 + batchbisect(commits[:l]) + batchbisect(commits[l:])
    
    return 1


def runbatch(batch):

    if 0 in batch:
        return False
    return True

def batchdivide4(batch):
    test_number =0
    global testNumber
    
    if (len(batch) >=20):
        sub_batch_len = len(batch)//4
        sub_batch_1 = batch[:sub_batch_len]
        sub_batch_2 = batch[sub_batch_len:2*sub_batch_len]
        sub_batch_3 = batch[2*sub_batch_len:-sub_batch_len]
        sub_batch_4 = batch[-sub_batch_len:]
        
        #print('hello {} {} {} {}'.format(len(sub_batch_1), len(sub_batch_2), len(sub_batch_3), len(sub_batch_4)))
        
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4
        
    
    elif (len(batch) == 12):
        sub_batch_1 = batch[0:3]
        sub_batch_2 = batch[3:6]
        sub_batch_3 = batch[6:9]
        sub_batch_4 = batch[9:12]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4


    elif (len(batch) == 11):
        sub_batch_1 = batch[0:3]
        sub_batch_2 = batch[3:6]
        sub_batch_3 = batch[6:9]
        sub_batch_4 = batch[9:11]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4



    elif (len(batch) == 13):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:7]
        sub_batch_3 = batch[7:10]
        sub_batch_4 = batch[10:13]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4


    elif (len(batch) == 14):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:7]
        sub_batch_3 = batch[7:11]
        sub_batch_4 = batch[11:14]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4



    elif (len(batch) == 15):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:8]
        sub_batch_3 = batch[8:12]
        sub_batch_4 = batch[12:15]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4



    elif (len(batch) == 16):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:8]
        sub_batch_3 = batch[8:12]
        sub_batch_4 = batch[12:16]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4




    elif (len(batch) == 17):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:8]
        sub_batch_3 = batch[8:12]
        sub_batch_4 = batch[12:17]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4



    elif (len(batch) == 18):
        sub_batch_1 = batch[0:5]
        sub_batch_2 = batch[5:9]
        sub_batch_3 = batch[9:14]
        sub_batch_4 = batch[14:18]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4



    elif (len(batch) == 19):
        sub_batch_1 = batch[0:5]
        sub_batch_2 = batch[5:9]
        sub_batch_3 = batch[9:14]
        sub_batch_4 = batch[14:19]
        if (runbatch(sub_batch_1) == False):
            test_number += batchstop4(sub_batch_1)
        if (runbatch(sub_batch_2) == False):
            test_number += batchstop4(sub_batch_2)
        if (runbatch(sub_batch_3) == False):
            test_number += batchstop4(sub_batch_3)
        if (runbatch(sub_batch_4) == False):
            test_number += batchstop4(sub_batch_4)
        test_number += 4



    elif (len(batch) <= 10):
        test_number+=batchstop4(batch)

    return test_number