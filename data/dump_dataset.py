with open('sentences.txt', 'r') as f: 
    with open('smallest_dataset.txt', 'w') as w: 
        for n, i in enumerate(f): 
            if n > 1: 
                break 
            else: 
                w.write(i)
