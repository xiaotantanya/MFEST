MarketArray=("ChinaA" "NASDAQ") #  
# ModelArray=("ALSTM" "EIIE" "GAT"  "SARL", "SFM", "DLinear")  
# ModelArray=("HSTGM" "RAT" "DeepTrader" "MGCGRU" "PPN") # "HSTGM" "RAT" "DeepTrader" "MGCGRU" "PPN"
ModelArray=("MASTER") # "HSTGM" "RAT" "DeepTrader" "MGCGRU" "PPN" "MFEST_WF" "MFEST_WIG" "MFEST_WSG"  AlphaPortfolio MASTER StockMixer
length=${#MarketArray[@]}
mlength=${#ModelArray[@]}
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 
# 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 
# 50 51 52 53 54 55 56 57 58 59 
for MODEL_INDEX in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 
do
for ((i=0; i<length; i++))
do
for ((j=0; j<mlength; j++))
do
python -u main.py \
--market ${MarketArray[i]} \
--model ${ModelArray[j]} \
--index ${MODEL_INDEX} \
--device "3"
done
done
done