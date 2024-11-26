import numpy


def PktSecFeature(times, sizes, features, howlong):
    bucket_num = 20 # 아래쪽에 있던 건데 예외 처리 때문에 위로 땡겨왔음

    if len(times)==0 or not any(sizes):
        features += [0] * (howlong+bucket_num+6)
        return;

    count = [0] * howlong
    
    for i in range(0, len(sizes)):
        t = int(numpy.floor(times[i]))
        if t < howlong:
            count[t] = count[t] + 1
    #except: print(len(times), len(sizes), t, howlong, times)
    features.extend(count)

    # mean, standard deviation, min, max, median
    features.append(numpy.mean(count))
    features.append(numpy.std(count))
    features.append(numpy.min(count))
    features.append(numpy.max(count))
    features.append(numpy.median(count))

    # alternative: 20 buckets
    bucket = [0] * bucket_num
    for i in range(0, howlong):
        ib = i // (howlong // bucket_num)
        bucket[ib] = bucket[ib] + count[i]
    features.extend(bucket)
    features.append(numpy.sum(bucket))
