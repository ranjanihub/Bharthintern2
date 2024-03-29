#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#dataset
titanic = pd.read_csv(r"C:\Users\TRINITY ELE\Downloads\titanic\train.csv")
print(titanic.head())

print(titanic.info())

print(titanic.describe())
print(titanic.isnull().sum())

# Visualize missing values
sns.heatmap(titanic.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Data')
plt.show()


# In[6]:


titanic['Age'].fillna(titanic['Age'].median(), inplace=True)

titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

titanic.drop('Cabin', axis=1, inplace=True, errors='ignore')
print(titanic.head())

# Visualize distribution of variables
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=titanic, palette='RdBu_r')
plt.title('Survival Count')
plt.show()

# Survival count by gender
sns.countplot(x='Survived', hue='Sex', data=titanic, palette='RdBu_r')
plt.title('Survival Count by Gender')
plt.show()

# Survival count by passenger class
sns.countplot(x='Survived', hue='Pclass', data=titanic, palette='rainbow')
plt.title('Survival Count by Passenger Class')
plt.show()

# Age distribution by survival
sns.histplot(x='Age', hue='Survived', data=titanic, kde=True)
plt.title('Age Distribution by Survival')
plt.show()

# Fare distribution by class and survival
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=titanic, palette='rainbow')
plt.title('Fare Distribution by Class and Survival')
plt.show()

# Survival count by port of embarkation
sns.countplot(x='Survived', hue='Embarked', data=titanic, palette='Set2')
plt.title('Survival Count by Port of Embarkation')
plt.show()

# Correlation heatmap
corr = titanic.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

