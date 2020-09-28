# -*- coding: utf-8 -*-
"""
Created on Sun Sept  27 16:05:44 2020
@author: Darren
"""

import glassdoor_scraper as gs
import pandas as pd

path = "/Users/yolkify/Documents/Data Science Projects/ds_salary_proj/chromedriver"

df = gs.get_jobs('python developer', 1000, False, path, 15)

df.to_csv('glassdoor_jobs.csv', index=False)
