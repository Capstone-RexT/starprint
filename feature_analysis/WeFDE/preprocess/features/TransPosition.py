import numpy
from features.common import X

CNT = 100 # 300:nonslot,  # 100: slot

# Transpositions (similar to good distance scheme)
# how many packets are in front of the outgoing/incoming packet?
def TransPosFeature(times, sizes, features):
    if len(times)==0 or not any(times):
        features += [0] * (2*CNT+4)
        return; 

    # for outgoing packets
    count = 0
    temp = []
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i)
            temp.append(i)
        if count == CNT:
            break
    for i in range(count, CNT):
        features.append(X)

    if len(temp) == 0:
        features.append(0)
        features.append(0)
    else:
        # std
        features.append(numpy.std(temp))
        # ave
        features.append(numpy.mean(temp))

    # for incoming packets
    count = 0
    temp = []
    for i in range(0, len(sizes)):
        if sizes[i] < 0:
            count += 1
            features.append(i)
            temp.append(i)
        if count == CNT:
            break
    for i in range(count, CNT):
        features.append(X)
    
    # 예외처리 by SJ
    if len(temp) == 0:
        features.append(0)
        features.append(0)
    else:
        # std
        features.append(numpy.std(temp))
        # ave
        features.append(numpy.mean(temp))
