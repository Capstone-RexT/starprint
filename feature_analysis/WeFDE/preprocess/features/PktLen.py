def PktLenFeature(times, sizes, features):
    for i in range(-1500, 1501):
        if i in sizes:
            features.append(1)
            #print("**")
        else:
            features.append(0)
            #print("^^")
        
