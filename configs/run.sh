
for file in $(ls */*.py); do
	(cat $file  && echo " \nprint(model['type'])" )| python; 
done
