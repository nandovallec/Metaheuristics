mv *iris*10*.csv 111.csv
mv *ecoli*10*.csv 222.csv
mv *rand*10*.csv 333.csv
mv *newthyroid*10*.csv 444.csv




tail -n 1 111.csv | awk -F'[: j]' '{ printf "%s %s %s ", $11, $8, $5 }' && tail -n 2 111.csv | head -n 1 | awk -F'[,: ]' '{printf "%s ", $4}'

tail -n 1 222.csv | awk -F'[: j]' '{ printf "%s %s %s ", $11, $8, $5 }' && tail -n 2 222.csv | head -n 1 | awk -F'[,: ]' '{printf "%s ", $4}'

tail -n 1 333.csv | awk -F'[: j]' '{ printf "%s %s %s ", $11, $8, $5 }' && tail -n 2 333.csv | head -n 1 | awk -F'[,: ]' '{printf "%s ", $4}'

tail -n 1 444.csv | awk -F'[: j]' '{ printf "%s %s %s ", $11, $8, $5 }' && tail -n 2 444.csv | head -n 1 | awk -F'[,: ]' '{print $4}'
echo "x"

