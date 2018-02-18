import csv
import numpy as np 
from numpy import genfromtxt
import scipy.stats as stats
import math

					#opening data sets into arrays

csv = np.genfromtxt('fever.csv', delimiter=',', dtype=None)
fever_data = csv[1:,:]

csv = np.genfromtxt('WBC.csv', delimiter=',', dtype=None)
blood_data = csv[1:,:]
			

					#splitting up data into diseases and disease-free

fever_none = np.array([])
fever_py27 = np.array([])
fever_py3 = np.array([])

blood_none = np.array([])
blood_py27 = np.array([])
blood_py3 = np.array([])

i = -1
for e in fever_data[:,0]:
	i += 1
	if e == 'No':
		number = float(fever_data[i,1])
		fever_none = np.append(fever_none, number)
	elif e == 'Py2.7':
		number = float(fever_data[i,1])
		fever_py27 = np.append(fever_py27, number)
	elif e == 'Py3':
		number = float(fever_data[i,1])
		fever_py3 = np.append(fever_py3, number)

i = -1
for e in blood_data[:,0]:
	i += 1
	if e == 'No':
		number = float(blood_data[i,1])
		blood_none = np.append(blood_none, number)
	elif e == 'Py2.7':
		number = float(blood_data[i,1])
		blood_py27 = np.append(blood_py27, number)
	elif e == 'Py3':
		number = float(blood_data[i,1])
		blood_py3 = np.append(blood_py3, number)

					#two sample t-test for statistical significance

#assumes samples have different variances
print stats.ttest_ind(fever_none, fever_py27, equal_var=False)
print stats.ttest_ind(fever_none, fever_py3, equal_var=False)
print stats.ttest_ind(fever_py27, fever_py3, equal_var=False)
print '-----------------------------------------------'
print stats.ttest_ind(blood_none, blood_py27, equal_var=False)
print stats.ttest_ind(blood_none, blood_py3, equal_var=False)
print stats.ttest_ind(blood_py27, blood_py3, equal_var=False)
print '-----------------------------------------------' 

					#calulating effect size

def hedges_g(sample1, sample2):
	n1, n2 = float(len(sample1)), float(len(sample2))
	s1, s2 = np.std(sample1), np.std(sample2)
	x1, x2 = sample1.mean(), sample2.mean()

	s_pooled = math.sqrt( ((n1-1)*s1**2 + (n2-1)*s2**2 ) / (n1+n2-2))
	cohens_d = (x1 - x2) / s_pooled
	hedges_g = cohens_d * (1- ((3) / (4*(n1+n2)-9)))
	return hedges_g

print hedges_g(fever_py27, fever_none)
print hedges_g(fever_py3, fever_none)
print hedges_g(fever_py27, fever_py3)
print '-----------------------------------------------'
print hedges_g(blood_none, blood_py27)
print hedges_g(blood_none, blood_py3)
print hedges_g(blood_py3, blood_py27)

					#Conditional probability

p_fgivend = 19.0/49
p_d = 24.0/69
p_f = 29.0/89

p_dgivenf = (p_fgivend * p_d)/p_f * 100
print '-----------------------------------------------'
print p_dgivenf 
