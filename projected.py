# ============================================================
# EMPLOYEE DATA ANALYSIS PROJECT
# Author: Abdullah Khan
# Description: Exploratory Data Analysis using Pandas
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES AND LOAD DATASET
# ============================================================

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1206)
pd.set_option('display.max_rows', 10)

dash = pd.read_csv(r"C:\Users\Abdullah\Desktop\Only Python\employee_data_analysis_practice.csv")


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n--- Dataset Shape ---")
print(dash.shape)

print("\n--- Total Employees ---")
print(dash['Employee_Name'].count())

print("\n--- Column Data Types ---")
print(dash.dtypes)

print("\n--- Dataset Information ---")
dash.info()

print("\n--- Total Missing Values ---")
print(dash.isnull().sum().sum())

print("\n--- Missing Values in Each Column ---")
print(dash.isnull().sum())

print("\n--- Duplicate Rows ---")
print(dash.duplicated().sum())

print("\n--- Unique Employee Names ---")
print(dash['Employee_Name'].nunique())

print("\n--- Statistical Summary ---")
print(dash.describe())

print("\n--- Unique Values in Each Column ---")
print(dash.nunique())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove complete duplicate rows
dash.drop_duplicates(inplace=True)

# Convert Joining_Date into datetime format
dash['Joining_Date'] = pd.to_datetime(
    dash['Joining_Date'], errors='coerce'
)

# Numeric columns: fill missing values using median
numeric_cols = dash.select_dtypes(include='number').columns

dash[numeric_cols] = dash[numeric_cols].fillna(
    dash[numeric_cols].median()
)

# Categorical columns: fill missing values with Unknown
categorical_cols = dash.select_dtypes(
    include='object'
).columns.drop('Attrition', errors='ignore')

dash[categorical_cols] = dash[categorical_cols].fillna('Unknown')

# Keep missing Attrition values as missing.
# They should not be treated as Yes or No.

print("\n--- Missing Values After Cleaning ---")
print(dash.isnull().sum())


# ============================================================
# 4. DEPARTMENT AND EMPLOYEE ANALYSIS
# ============================================================

print("\n--- Employees in Each Department ---")

deps = (
    dash.groupby('Department')['Employee_ID']
    .count()
    .sort_values(ascending=False)
    .reset_index(name='Employee_Count')
)

print(deps)

print("\n--- Department with Maximum Employees ---")
print(deps.head(1))

print("\n--- Department with Minimum Employees ---")
print(deps.tail(1))


print("\n--- Employees in Each Job Role ---")

roles = (
    dash.groupby('Job_Role')['Employee_ID']
    .count()
    .sort_values(ascending=False)
    .reset_index(name='Employee_Count')
)

print(roles)


print("\n--- Average Age by Department ---")

aves = (
    dash.groupby('Department')['Age']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(aves)


print("\n--- Department-wise Employee Percentage ---")

force = (
    dash['Department'].value_counts(normalize=True) * 100
)

print(force.sort_values(ascending=False))


print("\n--- Employees in Each City ---")

emps = (
    dash.groupby('City')['Employee_ID']
    .count()
    .sort_values(ascending=False)
    .reset_index(name='Employee_Count')
)

print(emps)


# ============================================================
# 5. SALARY ANALYSIS
# ============================================================

print("\n--- Overall Salary Statistics ---")

print("Maximum Salary:", dash['Annual_Salary_INR'].max())
print("Minimum Salary:", dash['Annual_Salary_INR'].min())
print("Average Salary:", dash['Annual_Salary_INR'].mean())
print("Median Salary:", dash['Annual_Salary_INR'].median())


print("\n--- Top 10 Highest-Paid Employees ---")

paid = dash.sort_values(
    'Annual_Salary_INR', ascending=False
).head(10)

print(
    paid[['Department', 'Annual_Salary_INR', 'Employee_Name']]
)


print("\n--- Average Salary by Department ---")

pay = (
    dash.groupby('Department')['Annual_Salary_INR']
    .mean()
    .sort_values(ascending=False)
)

print(pay)

print("\nHighest Average Salary Department:")
print(pay.head(1))

print("\nLowest Average Salary Department:")
print(pay.tail(1))


print("\n--- Total Salary by Department ---")

pays = (
    dash.groupby('Department')['Annual_Salary_INR']
    .sum()
    .sort_values(ascending=False)
)

print(pays)


print("\n--- Department-wise Contribution to Total Salary ---")

total = dash['Annual_Salary_INR'].sum()

sals = dash.groupby('Department')['Annual_Salary_INR'].sum()

total_sales = (
    (sals / total * 100)
    .sort_values(ascending=False)
    .reset_index(name='Salary_Contribution_Percentage')
)

print(total_sales)


print("\n--- Average Salary by Education ---")

sal_aves = (
    dash.groupby('Education')['Annual_Salary_INR']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(sal_aves)


print("\n--- Average Salary by Gender ---")

gens = (
    dash.groupby('Gender')['Annual_Salary_INR']
    .mean()
    .sort_values(ascending=False)
)

print(gens)


print("\n--- Employees Above and Below Average Salary ---")

avg = dash['Annual_Salary_INR'].mean()

print("Average Salary:", avg)
print("Below Average:", (dash['Annual_Salary_INR'] < avg).sum())
print("Above Average:", (dash['Annual_Salary_INR'] > avg).sum())


# Salary categories
dash['Salary_range'] = np.select(
    [
        dash['Annual_Salary_INR'] < 1000000,
        dash['Annual_Salary_INR'] <= 2000000
    ],
    [
        'Low',
        'Medium'
    ],
    default='High'
)

print("\n--- Salary Categories ---")

print(
    dash[['Annual_Salary_INR', 'Salary_range']].head(10)
)

print("\nSalary Category Counts:")
print(dash['Salary_range'].value_counts())


# ============================================================
# 6. AGE AND EXPERIENCE ANALYSIS
# ============================================================

print("\n--- Age Statistics by Department ---")

ags = (
    dash.groupby('Department')['Age']
    .agg(['min', 'max', 'mean'])
)

print(ags)


# Age groups
dash['Age_range'] = pd.cut(
    dash['Age'],
    bins=[0, 25, 30, 40, 50, float('inf')],
    labels=['20-25', '26-30', '31-40', '41-50', '51+']
)

print("\n--- Employee Count by Age Group ---")

print(dash['Age_range'].value_counts().sort_index())


print("\n--- Average Experience by Department ---")

exp = (
    dash.groupby('Department')['Experience_Years']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(exp)


print("\n--- Employees by Experience ---")

print(
    "More than 5 years:",
    (dash['Experience_Years'] > 5).sum()
)

print(
    "More than 10 years:",
    (dash['Experience_Years'] > 10).sum()
)

print(
    "More than 15 years:",
    (dash['Experience_Years'] > 15).sum()
)


print("\n--- Relationship Between Experience and Salary ---")

print(
    dash['Experience_Years'].corr(
        dash['Annual_Salary_INR']
    )
)


print("\n--- Average Salary by Age ---")

age_sals = (
    dash.groupby('Age')['Annual_Salary_INR']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(age_sals.head(10))


print("\n--- Average Experience by Job Role ---")

job = (
    dash.groupby('Job_Role')['Experience_Years']
    .mean()
    .sort_values(ascending=False)
)

print(job)


# ============================================================
# 7. PERFORMANCE AND TRAINING ANALYSIS
# ============================================================

print("\n--- Available Performance Ratings ---")
print(dash['Performance_Rating'].count())


print("\n--- Average Performance by Department ---")

high_perfos = (
    dash.groupby('Department')['Performance_Rating']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(high_perfos)


print("\n--- Employees with Excellent Rating ---")

print(
    "Rating 5:",
    (dash['Performance_Rating'] == 5).sum()
)


print("\n--- Average Salary by Performance Rating ---")

print(
    "Rating 1-2:",
    dash.loc[
        dash['Performance_Rating'].between(1, 2),
        'Annual_Salary_INR'
    ].mean()
)

print(
    "Rating 4-5:",
    dash.loc[
        dash['Performance_Rating'].between(4, 5),
        'Annual_Salary_INR'
    ].mean()
)


print("\n--- Relationship Between Training and Performance ---")

print(
    dash['Training_Hours'].corr(
        dash['Performance_Rating']
    )
)


print("\n--- Average Job Satisfaction by Department ---")

satis_job = (
    dash.groupby('Department')['Job_Satisfaction']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(satis_job)


print("\n--- Job Satisfaction Distribution ---")

print(
    "Dissatisfied:",
    dash['Job_Satisfaction'].between(1, 2).sum()
)

print(
    "Neutral:",
    (dash['Job_Satisfaction'] == 3).sum()
)

print(
    "Satisfied:",
    dash['Job_Satisfaction'].between(4, 5).sum()
)


# ============================================================
# 8. EMPLOYEE ATTRITION ANALYSIS
# ============================================================

# Exclude rows where Attrition is missing
attrition_data = dash.dropna(subset=['Attrition'])

print("\n--- Overall Attrition Count ---")

print(attrition_data['Attrition'].value_counts())


print("\n--- Overall Attrition Percentage ---")

attri = (
    attrition_data['Attrition']
    .value_counts(normalize=True) * 100
)

print(attri)


print("\n--- Attrition Count by Department ---")

ats = (
    attrition_data.groupby('Department')['Attrition']
    .value_counts()
    .reset_index(name='Employee_Count')
)

print(ats)


print("\n--- Attrition Rate by Department ---")

print(
    attrition_data.groupby('Department')['Attrition']
    .apply(lambda x: (x == 'Yes').mean() * 100)
    .sort_values(ascending=False)
)


print("\n--- Attrition Rate by Age Group ---")

lats = (
    attrition_data.groupby('Age_range', observed=True)['Attrition']
    .apply(lambda x: (x == 'Yes').mean() * 100)
    .sort_values(ascending=False)
    .reset_index(name='Attrition_Percentage')
)

print(lats)


print("\n--- Attrition Rate by Overtime ---")

slat = (
    attrition_data.groupby('Overtime')['Attrition']
    .apply(lambda x: (x == 'Yes').mean() * 100)
    .sort_values(ascending=False)
    .reset_index(name='Attrition_Percentage')
)

print(slat)


print("\n--- Average Job Satisfaction by Attrition ---")

sats = (
    attrition_data.groupby('Attrition')['Job_Satisfaction']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(sats)


print("\n--- Average Salary by Attrition ---")

attri_salary = (
    attrition_data.groupby('Attrition')['Annual_Salary_INR']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

print(attri_salary)


print("\n--- Attrition Rate by Experience ---")

salt = (
    attrition_data.groupby('Experience_Years')['Attrition']
    .apply(lambda x: (x == 'Yes').mean() * 100)
    .sort_values(ascending=False)
    .reset_index(name='Attrition_Percentage')
)

print(salt)


print("\n--- Attrition: Employees with Less Than 2 Years Experience ---")

less_2 = attrition_data[
    attrition_data['Experience_Years'] < 2
]

print(
    less_2['Attrition'].value_counts(normalize=True) * 100
)


print("\n--- Attrition: Employees with More Than 5 Years Experience ---")

more_5 = attrition_data[
    attrition_data['Experience_Years'] > 5
]

print(
    more_5['Attrition'].value_counts(normalize=True) * 100
)


print("\n--- Attrition by Promotion in Last 5 Years ---")

tals = pd.crosstab(
    attrition_data['Promotion_Last_5_Years'],
    attrition_data['Attrition'],
    normalize='index'
) * 100

print(tals)


# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================

print("\n--- Numerical Correlation Matrix ---")

print(
    dash.select_dtypes(include='number').corr()
)


# ============================================================
# 10. FINAL DATASET CHECK
# ============================================================

print("\n--- Final Dataset Shape ---")
print(dash.shape)

print("\n--- Remaining Missing Values ---")
print(dash.isnull().sum())

print("\n--- Final Dataset Preview ---")
print(dash.head())