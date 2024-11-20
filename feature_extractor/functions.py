def get_data(line):
    timestamp, size = map(float, line.split('\t'))
    dir = int(abs(size) / size)
    size = abs(size)
    return timestamp, dir, size

def get_count(instance):
    return len(instance)

def get_size(instance):    
    dirs = []
    for line in map(str.strip, instance):
        _, _, size = get_data(line)
        dirs.append(size)
    return dirs

def get_direction(instance):
    dirs = []
    for line in map(str.strip, instance):
        _, dir, _ = get_data(line)
        dirs.append(dir)
    return dirs

def get_tiktok(instance):
    feature = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = get_data(line)
        feature.append(timestamp * dir)
    return feature

def get_tam1d(instance):
    max_matrix_len = 1800
    maximum_load_time = 80
    timestamps = []
    dirs = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = get_data(line)
        timestamps.append(timestamp)
        dirs.append(dir)
    if timestamps:
        data = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
        for i in range(0, len(dirs)):
            if dirs[i] > 0:
                if timestamps[i] >= maximum_load_time:
                    data[0][-1] += 1
                else:
                    idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                    data[0][idx] += 1
            if dirs[i] < 0:
                if timestamps[i] >= maximum_load_time:
                    data[1][-1] += 1
                else:
                    idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                    data[1][idx] += 1
        return data[0] + data[1]
    return []

def get_ipd(instance):
    timestamps = []
    dirs = []
    first = True
    for line in map(str.strip, instance):
        timestamp, dir, _ = get_data(line)
        if first == True:
            first = False
            timestamps.append(0)
        else:
            timestamps.append(timestamp-beforetimestamp)
        dirs.append(dir)
        beforetimestamp = timestamp
    return timestamps