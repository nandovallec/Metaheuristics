mv *10_*123*.csv 111.csv
mv *10_*456*.csv 222.csv
mv *10_*789*.csv 333.csv
mv *10_*101112*.csv 444.csv
mv *10_*131415*.csv 555.csv

mv *20_*123*.csv 666.csv
mv *20_*456*.csv 777.csv
mv *20_*789*.csv 888.csv
mv *20_*101112*.csv 990.csv
mv *20_*131415*.csv 999.csv

tail -n 2 111.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 111.csv | awk -F': ' '{ print $2 }'
tail -n 2 222.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 222.csv | awk -F': ' '{ print $2 }'
tail -n 2 333.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 333.csv | awk -F': ' '{ print $2 }'
tail -n 2 444.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 444.csv | awk -F': ' '{ print $2 }'
tail -n 2 555.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 555.csv | awk -F': ' '{ print $2 }'
echo "#########################"
tail -n 2 666.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 666.csv | awk -F': ' '{ print $2 }'
tail -n 2 777.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 777.csv | awk -F': ' '{ print $2 }'
tail -n 2 888.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 888.csv | awk -F': ' '{ print $2 }'
tail -n 2 990.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 990.csv | awk -F': ' '{ print $2 }'
tail -n 2 999.csv | head -n 1 | awk -F'[,: ]' '{ printf "%s %s ",$4,$2}' && tail -n 1 999.csv | awk -F': ' '{ print $2 }'

