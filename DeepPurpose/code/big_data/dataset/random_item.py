#ylshi
import random
from random import randint
 
oldf=open('drug_protein.txt','r',encoding='UTF-8')
newf=open('drug_protein_out.txt','w',encoding='UTF-8')
n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.Random(0).sample(range(0,589578),5896)

lines=oldf.readlines()
for i in resultList:
    newf.write(lines[i])
    
oldf.close()
newf.close()


