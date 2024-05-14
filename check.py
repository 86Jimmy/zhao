# 判断两个文档之间的差异

def check(txt1, txt2):      
    fh1 = open(txt1, 'r') 
    fh2 = open(txt2, 'r')                 
    for line1, line2 in fh1, fh2:
        line1, line2 = line1.rstrip(), line2.rstrip()
        words1, words2 = line1.split(), line2.split()
                        


