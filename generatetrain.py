import random
# with open("traintestset_eng/train_files_eng.txt","r") as f:
#     t = f.readlines()
#     result = []
#     while len(result) < 13100:
#         temp = random.randint(0,13099)
#         if temp not in result:
#             result.append(temp)
#     fresult = []
#     for num in range(0,45):
#         tempresult = []
#         for i,j in enumerate(result[num*50:(num+1)*50]):
#             tempresult.append(t[j])
#         with open("traintestset_eng/train_files%s_eng.txt"%(num+1),"w") as f1:
#             f1.writelines(tempresult)
#     tempresult = []
#     for i,j in enumerate(result[-5000:]):
#         tempresult.append(t[j])
#     with open("traintestset_eng/test_files_eng.txt","w") as f1:
#         f1.writelines(tempresult)
with open("traintestset_chn/train_files.txt","r") as f:
    t = f.readlines()
    result = []
    while len(result) < 10000:
        temp = random.randint(0,9999)
        if temp not in result:
            result.append(temp)
    fresult = []
    for num in range(0,45):
        tempresult = []
        for i,j in enumerate(result[num*90:(num+1)*90]):
            tempresult.append(t[j])
        with open("traintestset_chn/train_files%s.txt"%(num+1),"w") as f1:
            f1.writelines(tempresult)
    tempresult = []
    for i,j in enumerate(result[-5000:]):
        tempresult.append(t[j])
    with open("traintestset_chn/test_files.txt","w") as f1:
        f1.writelines(tempresult)