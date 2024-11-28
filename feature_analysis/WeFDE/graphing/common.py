# Timing feature list
#FEATURE_CATEGORIES = [
#    ('Time', (1, 24)),
#    ('Pkt. per\n Second', (25, 150)),
#    ('IBD-LF', (151, 170)),
#    ('IBD-OFF', (171, 190)),
#    ('IBD-FF', (191, 210)),
#    ('IMD', (211, 230)),
#    ('Variance', (231, 250)),
#    ('MED', (251, 270)),
#    ('Burst\nLength', (271, 290)),
#    ('IBD-IFF', (291, 310)),
#]

# WeFDE feature list
'''1'''
# # {"PACKET_NUMBER": 13, "PKT_TIME": 37, "NGRAM": 161, "TRANS_POSITION": 225, "INTERVAL_KNN": 825, "INTERVAL_ICICS": 1427, "INTERVAL_WPES11": 2013, "PKT_DISTRIBUTION": 2238, "BURST": 2249, "FIRST20": 2269, "FIRST30_PKT_NUM": 2271, "LAST30_PKT_NUM": 2273, "PKT_PER_SECOND": 2399, "CUMUL": 2503}
'''
FEATURE_CATEGORIES = [
    ('Pkt. Count', (1, 13)), # 13
    ('Time', (14, 37)), # 24
    ('Ngram', (38, 161)), # (2**2+2**3+...+2**6=124)
    ('Transposition', (162, 765)),
    ('Interval-I', (766, 1365)),
    ('Interval-II', (1366, 1967)),
    ('Interval-III', (1968, 2553)),
    ('Pkt. Distribution', (2554, 2778)),
    ('Burst', (2779, 2789)),
    ('First 20', (2790, 2809)),
    ('First 30', (2810, 2811)),
    ('Last 30', (2812, 2813)),
    ('Pkt. per Second', (2814, 2939)),
    ('CUMUL', (2940, 3043))
]
'''

'''2'''
{"PACKET_NUMBER": 13, "PKT_TIME": 37, "NGRAM": 161, "TRANS_POSITION": 365, "INTERVAL_KNN": 565, "INTERVAL_ICICS": 767, "INTERVAL_WPES11": 953, "PKT_DISTRIBUTION": 1178, 
"BURST": 1189, "FIRST20": 1209, "FIRST30_PKT_NUM": 1211, "LAST30_PKT_NUM": 1213, "PKT_PER_SECOND": 1339, "CUMUL": 1443}
FEATURE_CATEGORIES = [
    ('Pkt. Count', (1, 13)), # 13
    ('Time', (14, 37)), # 24
    ('Ngram', (38, 161)), # (2**2+2**3+...+2**6=124)
    ('Transposition', (162, 365)),
    ('Interval-KNN', (366, 565)),
    ('Interval-ICICS', (566, 767)),
    ('Interval-WPES11', (768, 953)),
    ('Pkt. Distribution', (954, 1178)),
    ('Burst', (1179, 1189)),
    ('First 20', (1190, 1209)),
    ('First 30', (1210, 1211)),
    ('Last 30', (1212, 1213)),
    ('Pkt. per Second', (1214, 1339)),
    ('CUMUL', (1340, 1443))
]

#{"PACKET_NUMBER": 13, "PKT_TIME": 37, "TRANS_POSITION": 241, "BURST": 252, "CUMUL": 356}
# FEATURE_CATEGORIES = [
#     ('Pkt. Count', (1, 13)), # 13
#     ('Time', (14, 37)), # 24
#     ('Transposition', (38, 241)),
#     ('Burst', (242, 252)),
#     ('CUMUL', (253, 356))
# ]

# FEATURE_CATEGORIES = [
#     ('Pkt. Count', (1, 13)), # 13
#     ('Time', (14, 37)), # 24
#     ('Ngram', (38, 161)), # (2**2+2**3+...+2**6=124)
#     ('Transposition', (162, 765)),
#     ('Interval-I', (766, 1365)),
#     ('Interval-II', (1366, 1567)),
#     ('Interval-III', (1568, 1753)),
#     ('Pkt. Distribution', (1754, 1978)),
#     ('Burst', (1979, 1989)),
#     ('First 20', (1990, 2009)),
#     ('First 30', (2010, 2011)),
#     ('Last 30', (2012, 2013)),
#     ('Pkt. per Second', (2014, 2139)),
#     ('CUMUL', (2140, 2243))
# ]

# FEATURE_CATEGORIES = [
#     ('Pkt. Count', (1, 13)), # 13
#     ('Time', (14, 37)), # 24
#     ('Ngram', (38, 161)), # (2**2+2**3+...+2**6=124)
#     ('Transposition', (162, 225)),
#     ('Interval-I', (226, 825)),
#     ('Interval-II', (826, 1427)),
#     ('Interval-III', (1428, 2013)),
#     ('Pkt. Distribution', (2014, 2238)),
#     ('Burst', (2239, 2249)),
#     ('First 20', (2250, 2269)),
#     ('First 30', (2270, 2271)),
#     ('Last 30', (2272, 2273)),
#     ('Pkt. per Second', (2274, 2399)),
#     ('CUMUL', (2400, 2503))
# ]

# 5000-input DF features
#FEATURE_CATEGORIES = [
#    ('DF Features', (1, 5120))
#]
"""
Define names and ranges for feature categories.
"""


COLORS = [
    'blue',
    'green',
    'cyan',
    'red',
    'purple',
    'yellow',
    'olive',
    'orange',
    'violet'
    'teal',
]
"""
Ordered list of colors to use when graphing.
"""
