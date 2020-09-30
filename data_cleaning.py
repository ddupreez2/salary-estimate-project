"""
Created on Mon Sept 28 16:05:44 2020
@author: Darren du Preez
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

# salary parsing

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)

df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K', '').replace('CA$', ''))

min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2
 
# company name text only

df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

# state field

df['city'] = df['Location']

# age of company

df['age'] = df.Founded.apply(lambda x: x if x <1 else 2020 - x)

# parsing of job description (python, etc.)

#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df.python_yn.value_counts()

#html
df['html_yn'] = df['Job Description'].apply(lambda x: 1 if 'html' in x.lower() else 0)
df.html_yn.value_counts()

#css
df['css_yn'] = df['Job Description'].apply(lambda x: 1 if 'css' in x.lower() else 0)
df.css_yn.value_counts()

#aws
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws_yn.value_counts()

#django
df['django_yn'] = df['Job Description'].apply(lambda x: 1 if 'django' in x.lower() else 0)
df.django_yn.value_counts()

#postgres
df['postgresql_yn'] = df['Job Description'].apply(lambda x: 1 if 'postgresql' in x.lower() else 0)
df.postgresql_yn.value_counts()

df.to_csv('salary_data_cleaned.csv', index = False)