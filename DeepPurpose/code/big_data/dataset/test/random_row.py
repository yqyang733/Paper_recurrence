import random
import time
"""
注意盘符小写
"""
f = open("drug_protein.txt","r")           #源文件
fw = open("drug_protein_out.txt","w")     #待写文件
fr = open("drug_protein_rest.txt","w")
def test():  
    start = time.clock()
    raw_list = f.readlines()
    random.shuffle(raw_list)
    for i in range(5839):                           #随机抽取数目 n
        fw.writelines(raw_list[i])
    
    for m in range(583893):
        fr.writelines(raw_list[m])

    end = time.clock()
    print("cost time is %f" %(end - start))
 
if __name__ =="__main__":
    test()
