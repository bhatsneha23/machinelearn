import pandas as pd     
import seaborn as sns
from pandas_profiling import ProfileReport


titanic =  sns.load_dataset('titanic')
print(titanic.head())

prof = ProfileReport(titanic)
prof.to_file(output_file = 'output.html')  #it will create a html file in your current working directory