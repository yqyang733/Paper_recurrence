echo "fold,auc,aupr" >> fold_final.txt
for i in `seq 0 9`
do
  cd result_${i}
  grep epoch valid_markdowntable.txt|grep -v "#" > temp
  a=0
  b=0
  while read line
  do
    #echo ${line}
    auc=`echo ${line} | awk -F "|" '{print $3}'`
    aupr=`echo ${line} | awk -F "|" '{print $4}'`
    a=`echo "${a} + ${auc}" | bc`
    b=`echo "${b} + ${aupr}" | bc`
  done < temp
  echo -e "fold_${i},\c" >> ../fold_final.txt
  c=`echo "scale=4; ${a}/30" | bc |awk '{print 0.$1}'`
  echo -e "${c},\c" >> ../fold_final.txt
  echo "scale=4; ${b}/30" | bc |awk '{print 0.$1}'>> ../fold_final.txt
  cd ..
done
