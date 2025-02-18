import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  

df = pd.read_csv("Dataset/Skyserver.csv")  

print(df.head())  
print(df.info())  
print(df.describe())   

print(df.isnull().sum())  
print(df.duplicated().sum())  

col=[ 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run','camcol','field','redshift', 'plate', 'mjd', 'fiberid', "specobjid"]
plt.subplots(4,4,figsize=(15,15)) 
for i in range(len(col)):  
    plt.subplot(4,4,i+1)
    sns.histplot(df,x=col[i],hue="class",element="step")
    plt.xlabel(col[i])
plt.show()

plt.subplots(4,4,figsize=(15,15)) 
for i in  range(len(col)):  
    plt.subplot(4,4,i+1)
    sns.boxplot(df,x=col[i])
    plt.xlabel(col[i])
plt.show()

plt.figure(figsize=(10, 6))  
sns.heatmap(df[col].corr(), annot=True, annot_kws={'size':6}, fmt='.2f')  
plt.show()

sns.pairplot(data=df[['u', 'g', 'r', 'i', 'z','class']],hue='class')
plt.show()

df['class'].value_counts().plot(kind='bar')  
plt.show()
