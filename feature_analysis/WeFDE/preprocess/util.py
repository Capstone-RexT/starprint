import numpy as np

# extract params
FEATURE_EXT = ".features"
NORMALIZE_TRAFFIC = 0

# turn on/off debug:
PACKET_NUMBER = True
PKT_TIME = True
UNIQUE_PACKET_LENGTH = False
NGRAM_ENABLE = True
TRANS_POSITION = True
PACKET_DISTRIBUTION = True
BURSTS = True
FIRST20 = True
CUMUL = True
FIRST30_PKT_NUM = True
LAST30_PKT_NUM = True
PKT_PER_SECOND = True
INTERVAL_KNN = True
INTERVAL_ICICS = True
INTERVAL_WPES11 = True

# packet number per second, how many seconds to count?
howlong = 100

# n-gram feature
NGRAM = 3

# CUMUL feature number
featureCount = 100


# Python3 conversion of python2 cmp function
def cmp(a, b):
    return (a > b) - (a < b)


# normalize traffic
def normalize_traffic(times, sizes):
    # sort
    tmp = sorted(zip(times, sizes))

    times = [x for x, _ in tmp]
    sizes = [x for _, x in tmp]

    TimeStart = times[0]
    PktSize = 500

    # normalize time
    for i in range(len(times)):
        times[i] = times[i] - TimeStart

    # normalize size
    for i in range(len(sizes)):
        sizes[i] = (abs(sizes[i]) / PktSize) * cmp(sizes[i], 0)

    # flat it
    newtimes = list()
    newsizes = list()

    for t, s in zip(times, sizes):
        numCell = abs(s)
        oneCell = cmp(s, 0)
        for r in range(numCell):
            newtimes.append(t)
            newsizes.append(oneCell)

    return newtimes, newsizes

'''from extract_timing_feature'''
def extract_bursts(trace):
    bursts = []
    direction_counts = []
    direction = trace[0][1]
    burst = []
    count = 1
    for i, packet in enumerate(trace):
        if packet[1] != direction:
            bursts.append(burst)
            burst = [packet]
            direction_counts.append(count)
            direction = packet[1]
            count = 1
        else:
            burst.append(packet)
            count += 1
    bursts.append(burst)
    return bursts, direction_counts


def direction_counts(trace):
    counts = []
    direction = trace[0][1]
    count = 1
    for packet in trace:
        if packet[1] != direction:
            counts.append(count)
            direction = packet[1]
            count = 1
        else:
            count += 1
    return counts


def get_bin_sizes(feature_values, bin_input):
    bin_raw = []
    for v in feature_values.values():
        bin_raw.extend(v)
    bin_s = np.sort(bin_raw)
    bins = np.arange(0, 100 + 1, 100.0 / bin_input)

    final_bin = [np.percentile(bin_s, e) for e in bins]
    return final_bin


def slice_by_binsize(feature_values, bin_input):
    bin_for_all_instances = np.array(get_bin_sizes(feature_values, bin_input))
    d_new = {}
    for name, v in feature_values.iteritems():
        d_new[name] = [[] for _ in range(bin_input)]

        bin_indices = np.digitize(np.array(v),
                                  bin_for_all_instances[:bin_input],
                                  right=True)
        for i in range(bin_indices.size):
            if bin_indices[i] > bin_input:
                d_new[name][-1].append(v[i])
            elif bin_indices[i] == 0:
                d_new[name][0].append(v[i])
            else:
                d_new[name][bin_indices[i] - 1].append(v[i])
    return d_new


def get_statistics(feature_values, bin_input):
    sliced_dic = slice_by_binsize(feature_values, bin_input)
    bin_length = {
        key: [len(value) for value in values] for key, values in
        sliced_dic.iteritems()
    }
    return bin_length


def normalize_data(feature_values, bin_input):
    to_be_norm = get_statistics(feature_values, bin_input)
    normed = {
        key: [float(value) / sum(values) for value in values]
        if sum(values) > 0 else values
        for key, values in to_be_norm.iteritems()
    }
    return normed


def final_format_by_class(feature_values, bin_input):
    # norm_data = normalized_data(traces, bin_input)
    norm_data = get_statistics(feature_values, bin_input)
    final = {}
    for k in norm_data:
        c = k.split('-')[0]
        if c not in final:
            final[c] = [norm_data[k]]
        else:
            final[c].append(norm_data[k])
    return final


def padding_neural(feature_values):
    directed_neural = feature_values
    max_length = max(len(elements) for elements in directed_neural.values())
    print("Maximum Length", max_length)
    for key, value in directed_neural.iteritems():
        if len(value) < max_length:
            zeroes_needed = max_length - len(value)
            value += [0] * zeroes_needed

    return directed_neural

def intraBD_med(bursts):
    intra_burst_delays = []
    for burst in bursts:
        timestamps = [packet[0] for packet in burst]
        intra_burst_delays.append(np.median(timestamps))
    return intra_burst_delays


def inter_inramd(bursts):
    primary = intraBD_med(bursts)
    processed = [q-p for p, q in zip(primary[:-1], primary[1:])]

    return processed


def intra_burst_delay_var(bursts):
    intra_burst_delays = []
    for burst in bursts:
        timestamps = [packet[0] for packet in burst]
        intra_burst_delays.append(np.var(timestamps))
    return intra_burst_delays


def inter_burst_delay_first_first(bursts):
    timestamps = [float(burst[0][0]) for burst in bursts]

    return np.diff(timestamps).tolist()


def inter_burst_delay_incoming_first_first(bursts):
    incoming_bursts = [burst for burst in bursts if burst[0][1] == -1]
    timestamps = [float(burst[0][0]) for burst in incoming_bursts]
    return np.diff(timestamps).tolist()


def inter_burst_delay_last_first(bursts):
    timestamps_first = [float(burst[0][0]) for burst in bursts]
    timestamps_last = [float(burst[-1][0]) for burst in bursts]
    inter_burst_delays = [i-j for i, j in zip(timestamps_last,
                                              timestamps_first)]
    return inter_burst_delays


def inter_burst_delay_outgoing_first_first(bursts):
    outgoing_bursts = [burst for burst in bursts if burst[0][1] == 1]
    timestamps = [float(burst[0][0]) for burst in outgoing_bursts]
    return np.diff(timestamps).tolist()


def intra_interval(bursts):
    timestamps_first = [float(burst[0][0]) for burst in bursts]
    timestamps_last = [float(burst[-1][0]) for burst in bursts]
    interval = [i-j for i, j in zip(timestamps_last, timestamps_first)]
    return interval