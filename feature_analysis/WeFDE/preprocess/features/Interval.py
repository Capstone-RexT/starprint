# inflow interval (icics, knn)
from features.common import X

CNT = 100 # 300 # 100

def IntervalFeature(times, sizes, features, Category):
    if Category == 'KNN':
        if len(times)==0 or not any(sizes):
            features += [0] * (CNT*2)
            return; 

        # a list of first 300 intervals (KNN)
        # incoming interval
        count = 0
        prevloc = 0
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                count += 1
                features.append(i - prevloc)
                prevloc = i
            if count == CNT:
                break
        for i in range(count, CNT):
            features.append(X)

        # outgoing interval
        count = 0
        prevloc = 0
        for i in range(0, len(sizes)):
            if sizes[i] < 0:
                count += 1
                features.append(i - prevloc)
                prevloc = i
            if count == CNT:
                break
        for i in range(count, CNT):
            features.append(X)

    if Category == "ICICS" or Category == "WPES11":
        MAX_INTERVAL = 100
        if (len(times)==0 or not any(sizes)) and Category == "ICICS":
            features += [0] * (MAX_INTERVAL*2+2)
            return;
        elif (len(times)==0 or not any(sizes)) and Category == "WPES11":
            features += [0] * (2*(6+(MAX_INTERVAL+1)-14))
            return;
    
        # Distribution of the intervals
        # incoming interval
        count = 0
        prevloc = 0
        interval_freq_in = [0] * (MAX_INTERVAL + 1)
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                inv = i - prevloc - 1
                prevloc = i
                # record the interval
                if inv > MAX_INTERVAL:
                    inv = MAX_INTERVAL
                interval_freq_in[inv] += 1

        # outgoing interval
        count = 0
        prevloc = 0
        interval_freq_out = [0] * (MAX_INTERVAL + 1)
        for i in range(0, len(sizes)):
            if sizes[i] < 0:
                inv = i - prevloc - 1
                prevloc = i
                # record the interval
                if inv > MAX_INTERVAL:
                    inv = MAX_INTERVAL
                interval_freq_out[inv] += 1

        # ICICS: no grouping
        if Category == "ICICS":
            features.extend(interval_freq_in)
            features.extend(interval_freq_out)

        # WPES 11: 1, 2, 3-5, 6-8, 9-13, 14 (grouping)
        if Category == "WPES11":
            # incoming
            features.extend(interval_freq_in[0:3]) # 3
            features.append(sum(interval_freq_in[3:6])) # 1
            features.append(sum(interval_freq_in[6:9])) # 1
            features.append(sum(interval_freq_in[9:14])) # 1
            features.extend(interval_freq_in[14:]) # MAX_INTERVAL-14
            # outgoing
            features.extend(interval_freq_out[0:3])
            features.append(sum(interval_freq_out[3:6]))
            features.append(sum(interval_freq_out[6:9]))
            features.append(sum(interval_freq_out[9:14]))
            features.extend(interval_freq_out[14:])
