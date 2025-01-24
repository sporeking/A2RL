def abl_trace(trace: list, state_trace: list, end: int, obs_trace: list, StateNN):
    min_invalid_count = float('inf')  # 初始化最小不符合条件的数字总个数为无穷大
    min_split = [] # 初始化最小不符合条件的分割情况为空
    max_probability = 0.0
    
    if len(state_trace) == 1:
        return 0, trace
    
    if len(state_trace) >= 4:
        for i in range(1, len(trace) - 2):
            for j in range(i, len(trace) - 1):
                probability = 1.0
                split1 = trace[:i]
                split2 = trace[i:j+1]
                split3 = trace[j+1:len(trace) - 1]

                invalid_count = 0
                for k in range(len(split1)):
                    if split1[k] != state_trace[0]:
                        invalid_count += 1
                        # probability *= StateNN()
                for k in range(len(split2)):
                    if split2[k] != state_trace[1]:
                        invalid_count += 1
                for k in range(len(split3)):
                    if split3[k] != state_trace[2]:
                        invalid_count += 1

                if invalid_count < min_invalid_count:
                    min_invalid_count = invalid_count
                    # min_split.clear()
                    probability = 1.0
                    for k in range(len(split1)):
                        probability *= StateNN(obs_trace[k])[split1[k]]
                    for k in range(len(split2)):
                        probability *= StateNN(obs_trace[len(split1) + k])[split2[k]]
                    for k in range(len(split3)):
                        probability *= StateNN(obs_trace[len(split1) + len(split2) + k])[split3[k]]
                    max_probability = probability
                    min_split = [state_trace[0] for _ in split1] + [state_trace[1] for _ in split2] + [state_trace[2] for _ in split3] + [end]
                
                if invalid_count == min_invalid_count:
                    probability = 1.0
                    for k in range(len(split1)):
                        probability *= StateNN(obs_trace[k])[split1[k]]
                    for k in range(len(split2)):
                        probability *= StateNN(obs_trace[len(split1) + k])[split2[k]]
                    for k in range(len(split3)):
                        # print(len(obs_trace), len(split1), len(split2), len(split3), k, split3[k])
                        probability *= StateNN(obs_trace[len(split1) + len(split2) + k])[split3[k]]
                    if probability > max_probability:
                        min_split = [state_trace[0] for _ in split1] + [state_trace[1] for _ in split2] + [state_trace[2] for _ in split3] + [end]
                    
    if len(state_trace) == 3 or (len(state_trace) >= 3 and end == 4):   
        for i in range(1, len(trace) - 2):
            split1 = trace[:i]
            split2 = trace[i:len(trace)-1]

            invalid_count = 0
            for k in range(len(split1)):
                if split1[k] != state_trace[0]:
                    invalid_count += 1
            for k in range(len(split2)):
                if split2[k] != state_trace[1]:
                    invalid_count += 1

            if invalid_count < min_invalid_count:
                min_invalid_count = invalid_count
                probability = 1.0
                for k in range(len(split1)):
                    probability *= StateNN(obs_trace[k])[split1[k]]
                for k in range(len(split2)):
                    probability *= StateNN(obs_trace[len(split1) + k])[split2[k]]
                max_probability = probability
                min_split = [state_trace[0] for _ in split1] + [state_trace[1] for _ in split2] + [end]
            
            if invalid_count == min_invalid_count:
                probability = 1.0
                for k in range(len(split1)):
                    probability *= StateNN(obs_trace[k])[split1[k]]
                for k in range(len(split2)):
                    probability *= StateNN(obs_trace[len(split1) + k])[split2[k]]
                if probability > max_probability:
                    min_split = [state_trace[0] for _ in split1] + [state_trace[1] for _ in split2] + [end]

    if len(state_trace) == 2 or (len(state_trace) >= 2 and end == 4):
        split1 = trace[0:len(trace) - 1]
        invalid_count = 0
        for k in range(len(split1)):
            if split1[k] != state_trace[0]:
                invalid_count += 1
        if invalid_count < min_invalid_count:
            min_invalid_count = invalid_count
            probability = 1.0
            for i in range(len(split1)):
                probability *= StateNN(obs_trace[i])[split1[i]]
            max_probability = probability
            min_split = [state_trace[0] for _ in split1] + [end]
        if invalid_count == min_invalid_count:
            probability = 1.0
            for i in range(len(split1)):
                probability *= StateNN(obs_trace[i])[split1[i]]
            if probability > max_probability:
                min_split = [state_trace[0] for _ in split1] + [end]

    if trace[-1] != end:
        min_invalid_count += 1
    print("trace length: ", len(trace), ", wrong state num: ", min_invalid_count)
    return min_invalid_count, min_split
