batch_duration = 0

def batchstop4(commits, duration=[1,1,1]):
    
    l = len(commits)//2
    
    if 0 in commits:
        
        if len(commits) < 4:
            print("I added 1 here")
            return 1 + len(commits)
        
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

def batchdivide4(batch, duration=[1,1,1,1]):
    test_number =0
    global testNumber
    
    global batch_duration
    
    duration = [1]*len(batch)
    
    
    if (len(batch) >=20):
        sub_batch_len = len(batch)//4
        sub_batch_1 = batch[:sub_batch_len]
        sub_batch_2 = batch[sub_batch_len:2*sub_batch_len]
        sub_batch_3 = batch[2*sub_batch_len:-sub_batch_len]
        sub_batch_4 = batch[-sub_batch_len:]
        
        sub_dur_1 = duration[:sub_batch_len]
        sub_dur_2 = duration[sub_batch_len:2*sub_batch_len]
        sub_dur_3 = duration[2*sub_batch_len:-sub_batch_len]
        sub_dur_4 = duration[-sub_batch_len:]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        
        ##print('hello {} {} {} {}'.format(len(sub_batch_1), len(sub_batch_2), len(sub_batch_3), len(sub_batch_4)))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)
        
    
    elif (len(batch) == 12):
        sub_batch_1 = batch[0:3]
        sub_batch_2 = batch[3:6]
        sub_batch_3 = batch[6:9]
        sub_batch_4 = batch[9:12]
        
        sub_dur_1 = duration[0:3]
        sub_dur_2 = duration[3:6]
        sub_dur_3 = duration[6:9]
        sub_dur_4 = duration[9:12]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)


    elif (len(batch) == 11):
        sub_batch_1 = batch[0:3]
        sub_batch_2 = batch[3:6]
        sub_batch_3 = batch[6:9]
        sub_batch_4 = batch[9:11]
        
        sub_dur_1 = duration[0:3]
        sub_dur_2 = duration[3:6]
        sub_dur_3 = duration[6:9]
        sub_dur_4 = duration[9:11]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)



    elif (len(batch) == 13):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:7]
        sub_batch_3 = batch[7:10]
        sub_batch_4 = batch[10:13]
        
        sub_dur_1 = duration[0:4]
        sub_dur_2 = duration[4:7]
        sub_dur_3 = duration[7:10]
        sub_dur_4 = duration[10:13]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)


    elif (len(batch) == 14):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:7]
        sub_batch_3 = batch[7:11]
        sub_batch_4 = batch[11:14]
        
        sub_dur_1 = duration[0:4]
        sub_dur_2 = duration[4:7]
        sub_dur_3 = duration[7:11]
        sub_dur_4 = duration[11:14]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)



    elif (len(batch) == 15):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:8]
        sub_batch_3 = batch[8:12]
        sub_batch_4 = batch[12:15]
        
        sub_dur_1 = duration[0:4]
        sub_dur_2 = duration[4:8]
        sub_dur_3 = duration[8:12]
        sub_dur_4 = duration[12:15]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)



    elif (len(batch) == 16):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:8]
        sub_batch_3 = batch[8:12]
        sub_batch_4 = batch[12:16]
        
        sub_dur_1 = duration[0:4]
        sub_dur_2 = duration[4:8]
        sub_dur_3 = duration[8:12]
        sub_dur_4 = duration[12:16]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)




    elif (len(batch) == 17):
        sub_batch_1 = batch[0:4]
        sub_batch_2 = batch[4:8]
        sub_batch_3 = batch[8:12]
        sub_batch_4 = batch[12:17]
        
        sub_dur_1 = duration[0:4]
        sub_dur_2 = duration[4:8]
        sub_dur_3 = duration[8:12]
        sub_dur_4 = duration[12:17]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)



    elif (len(batch) == 18):
        sub_batch_1 = batch[0:5]
        sub_batch_2 = batch[5:9]
        sub_batch_3 = batch[9:14]
        sub_batch_4 = batch[14:18]
        
        sub_dur_1 = duration[0:5]
        sub_dur_2 = duration[5:9]
        sub_dur_3 = duration[9:14]
        sub_dur_4 = duration[14:18]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)



    elif (len(batch) == 19):
        sub_batch_1 = batch[0:5]
        sub_batch_2 = batch[5:9]
        sub_batch_3 = batch[9:14]
        sub_batch_4 = batch[14:19]
        
        sub_dur_1 = duration[0:5]
        sub_dur_2 = duration[5:9]
        sub_dur_3 = duration[9:14]
        sub_dur_4 = duration[14:19]
        
        batch_duration += sub_dur_1[-1]
        batch_duration += sub_dur_2[-1]
        batch_duration += sub_dur_3[-1]
        batch_duration += sub_dur_4[-1]
        
        #print('BD4 Bisected into {} {} {} {}'.format(sub_batch_1, sub_batch_2, sub_batch_3, sub_batch_4))
        
        test_number += batchstop4(sub_batch_1, sub_dur_1)
        test_number += batchstop4(sub_batch_2, sub_dur_2)
        test_number += batchstop4(sub_batch_3, sub_dur_3)
        test_number += batchstop4(sub_batch_4, sub_dur_4)



    elif (len(batch) <= 10):
        test_number+=batchstop4(batch, duration)

    return test_number