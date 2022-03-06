
from sklearn.model_selection import StratifiedShuffleSplit


split = StratifiedShuffleSplit(n_splits=1, test_size=.02)

for train_index, test_index in split.split(df_file, df_file['income_cat']):
    strat_train_set = df_file.loc[train_index] # retrive rows with these indces
    strat_test_set = df_file.loc[test_index] # retrive rows with these indces





# The number of each category like from 0 to 1.5 about some precentage
stratified = np.array(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
random = np.array(measure_income_compare(test_set))/len(test_set)
overall = np.array(measure_income_compare(df_file))/ len(df_file)




stratified.sort()
random.sort()
overall.sort()
compare_dict = {'overall': overall, 'stratified': stratified, 'random': random}
compare_df = pd.DataFrame(compare_dict)

# In table below we can see how stratified static method most similar to overall coverage of our dataset
# look also at the simple random Vs overall
compare_df['Stratified. %error'] = np.abs((compare_df['stratified'] - compare_df['overall']) * 100)
compare_df['Random. %error'] = np.abs((compare_df['random'] - compare_df['overall']) * 100)
compare_df