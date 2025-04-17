# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 查看原始数据
data = pd.read_csv('D:\Desktop\cs-test.csv')
print('原始数据概况')
data.info()

# Check the columns and the first few rows to understand the structure
data.head(), data.columns

# Fill SeriousDlqin2yrs based on the sum of the three columns
data['SeriousDlqin2yrs'] = data[['NumberOfTime30-59DaysPastDueNotWorse',
                             'NumberOfTimes90DaysLate',
                             'NumberOfTime60-89DaysPastDueNotWorse']].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# Check if the values are updated correctly
data[['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'SeriousDlqin2yrs']].head()

# 数据清洗函数：随机森林填充缺失值
def set_missing(df):
    print('随机森林回归填充0值：')
    process_df = df.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    known = process_df.loc[process_df['MonthlyIncome'] != 0].values
    unknown = process_df.loc[process_df['MonthlyIncome'] == 0].values
    X = known[:, 1:]
    y = known[:, 0]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=30, n_jobs=-1)
    rfr.fit(X, y)
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    df.loc[df['MonthlyIncome'] == 0, 'MonthlyIncome'] = predicted
    return df
# 数据清洗函数：删除异常值
def outlier_processing(df, cname):
    s = df[cname]
    onequater = s.quantile(0.25)
    threequater = s.quantile(0.75)
    irq = threequater - onequater
    min_val = onequater - 1.5 * irq
    max_val = threequater + 1.5 * irq
    df = df[df[cname] <= max_val]
    df = df[df[cname] >= min_val]
    return df


# MonthlyIncome属性离群点原始分布
print('MonthlyIncome属性离群点原始分布：')
data[['MonthlyIncome']].boxplot()
plt.savefig('MonthlyIncome1.png', dpi=300, bbox_inches='tight')
plt.show()

# 删除离群点并填充缺失值
print('删除离群点，填充缺失数据：')
data = outlier_processing(data, 'MonthlyIncome')
data = set_missing(data)
print('处理MonthlyIncome后数据概况：')
data.info()

# MonthlyIncome属性离群点处理后分布
data[['MonthlyIncome']].boxplot()
plt.savefig('MonthlyIncome2.png', dpi=300, bbox_inches='tight')
plt.show()

# 处理其他属性的离群点
data = outlier_processing(data, 'age')
data = outlier_processing(data, 'RevolvingUtilizationOfUnsecuredLines')
data = outlier_processing(data, 'DebtRatio')
data = outlier_processing(data, 'NumberOfOpenCreditLinesAndLoans')
data = outlier_processing(data, 'NumberRealEstateLoansOrLines')
data = outlier_processing(data, 'NumberOfDependents')

# 处理三个离群点分布过于集中的属性
features = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate']
features_labels = ['30-59days', '60-89days', '90+days']
print('三个属性的原始分布：')
data[features].boxplot()
plt.xticks([1, 2, 3], features_labels)
plt.savefig('三个属性的原始分布', dpi=300, bbox_inches='tight')
plt.show()

print('删除离群点后：')
data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
data = data[data['NumberOfTime60-89DaysPastDueNotWorse'] < 90]
data = data[data['NumberOfTimes90DaysLate'] < 90]

data[features].boxplot()
plt.xticks([1, 2, 3], features_labels)
plt.savefig('三个属性的整理后分布', dpi=300, bbox_inches='tight')
plt.show()

print('处理离群点后数据概况：')
data.info()

# 生成数据集和测试集
# 将SeriousDlqin2yrs进行0-1转换
#生成数据集和测试集
from sklearn.model_selection import train_test_split
#原始值0为正常，1为违约。因为习惯上信用评分越高，违约的可能越小，所以将原始值0和1置换
data['SeriousDlqin2yrs'] = 1-data['SeriousDlqin2yrs']
Y = data['SeriousDlqin2yrs']
X = data.iloc[:,1:]



#拆分训练集和数据集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state = 0)

train = pd.concat([Y_train,X_train],axis = 1)
test = pd.concat([Y_test,X_test],axis = 1)
clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
print('训练集数据')
print(train.shape)
print('测试集数据')
print(test.shape)


#对属性进行分箱，并计算WOE和IV值
def mono_bin(res,feat,n = 10):
    good = res.sum()
    bad = res.count()-good
    d1 = pd.DataFrame({'feat':feat,'res':res,'Bucket':pd.cut(feat,n)})
    d2 = d1.groupby('Bucket',as_index = True)
    d3 = pd.DataFrame(d2.feat.min(),columns = ['min'])
    d3['min'] = d2.min().feat
    d3['max'] = d2.max().feat
    d3['sum'] = d2.sum().res
    d3['total'] = d2.count().res
    d3['rate'] = d2.mean().res
    d3['woe'] = np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum']/good
    d3['badattribute'] = (d3['total']-d3['sum'])/bad
    iv = ((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min'))
    cut = []
    cut.append(float('-inf'))
    for i in range(1,n):
        qua = feat.quantile(i/(n))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe

def self_bin(res,feat,cat):
    good = res.sum()
    bad = res.count()-good
    d1 = pd.DataFrame({'feat':feat,'res':res,'Bucket':pd.cut(feat,cat)})
    d2 = d1.groupby('Bucket',as_index = True)
    d3 = pd.DataFrame(d2.feat.min(),columns = ['min'])
    d3['min'] = d2.min().feat
    d3['max'] = d2.max().feat
    d3['sum'] = d2.sum().res
    d3['total'] = d2.count().res
    d3['rate'] = d2.mean().res
    d3['woe'] = np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum']/good
    d3['badattribute'] = (d3['total']-d3['sum'])/bad
    iv = ((d3['goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min'))
    woe = list(d4['woe'].round(3))
    return d4,iv,woe

pinf = float('inf')
ninf = float('-inf')
dfx1,ivx1,cutx1,woex1 = mono_bin(train['SeriousDlqin2yrs'],train['RevolvingUtilizationOfUnsecuredLines'],n = 10)
#显示RevolvingUtilizationOfUnsecuredLines分箱和WOE信息
print('='*60)
print('显示RevolvingUtilizationOfUnsecuredLines分箱和WOE信息:')
print(dfx1)
dfx2,ivx2,cutx2,woex2 = mono_bin(train['SeriousDlqin2yrs'],train['age'],n = 10)
dfx4,ivx4,cutx4,woex4 = mono_bin(train['SeriousDlqin2yrs'],train['DebtRatio'],n = 10)
dfx5,ivx5,cutx5,woex5 = mono_bin(train['SeriousDlqin2yrs'],train['MonthlyIncome'],n = 10)
#对3，6，7，8，9，10列数据进行指定间隔分箱
cutx3 = [ninf,0,1,3,5,pinf]
cutx6 = [ninf,1,2,3,5,pinf]
cutx7 = [ninf,0,1,3,5,pinf]
cutx8 = [ninf,0,1,2,3,pinf]
cutx9 = [ninf,0,1,3,pinf]
cutx10 = [ninf,0,1,2,3,5,pinf]

#按照cutx3指定的间隔把NumberOfTime30-59DaysPastDueNotWorse属性分成5段
dfx3,ivx3,woex3 = self_bin(train['SeriousDlqin2yrs'],train['NumberOfTime30-59DaysPastDueNotWorse'],cutx3)
#显示NumberOfTime30-59DaysPastDueNotWorse分箱和woe信息：
print('='*60)
print('NumberOfTime30-59DaysPastDueNotWorse分箱和woe信息：')
print(dfx3)
dfx6,ivx6,woex6 = self_bin(train['SeriousDlqin2yrs'],train['NumberOfOpenCreditLinesAndLoans'],cutx6)
dfx7,ivx7,woex7 = self_bin(train['SeriousDlqin2yrs'],train['NumberOfTimes90DaysLate'],cutx7)
dfx8,ivx8,woex8 = self_bin(train['SeriousDlqin2yrs'],train['NumberRealEstateLoansOrLines'],cutx8)
dfx9,ivx9,woex9 = self_bin(train['SeriousDlqin2yrs'],train['NumberOfTime60-89DaysPastDueNotWorse'],cutx9)
dfx10,ivx10,woex10 = self_bin(train['SeriousDlqin2yrs'],train['NumberOfDependents'],cutx10)

#按照iv选取属性
ivlist = [ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9,ivx10]
index = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
x = np.arange(len(index))+1
ax1.bar(x,ivlist,width = 0.48,color = 'yellow',alpha = 0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(index,rotation = 0,fontsize = 12)
ax1.set_ylabel('IV(information value)',fontsize = 14)
for a,b in zip(x,ivlist):
    plt.text(a,b+0.01,'%.4f'%b,ha = 'center',va = 'bottom',fontsize = 10)
plt.savefig('iv取值.png', dpi = 300,bbox_inches = 'tight')
plt.show()

#模型训练阶段
#求出属性的对应woe值
def get_woe(feat, cut, woe):
    res = []
    for _, value in feat.items():  # 使用 items() 代替 iteritems()
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        res.append(woe[m])
    return res

def compute_score(feat, cut, score):
    res = []
    for _, value in feat.items():  # 使用 items() 代替 iteritems()
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        res.append(score[m])
    return res

#调用get_woe函数，分别将训练集和测试集的属性值转为woe值
woe_train = pd.DataFrame()
woe_train['SeriousDlqin2yrs'] = train['SeriousDlqin2yrs']
woe_train['RevolvingUtilizationOfUnsecuredLines'] = get_woe(train['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1)
woe_train['age'] = get_woe(train['age'], cutx2, woex2)
woe_train['NumberOfTime30-59DaysPastDueNotWorse'] = get_woe(train['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3)
woe_train['DebtRatio'] = get_woe(train['DebtRatio'], cutx4, woex4)
woe_train['MonthlyIncome'] = get_woe(train['MonthlyIncome'], cutx5, woex5)
woe_train['NumberOfOpenCreditLinesAndLoans'] = get_woe(train['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6)
woe_train['NumberOfTimes90DaysLate'] = get_woe(train['NumberOfTimes90DaysLate'], cutx7, woex7)
woe_train['NumberRealEstateLoansOrLines'] = get_woe(train['NumberRealEstateLoansOrLines'], cutx8, woex8)
woe_train['NumberOfTime60-89DaysPastDueNotWorse'] = get_woe(train['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9)
woe_train['NumberOfDependents'] = get_woe(train['NumberOfDependents'], cutx10, woex10)

#将测试集各属性替换成woe
woe_test = pd.DataFrame()
woe_test['SeriousDlqin2yrs'] = train['SeriousDlqin2yrs']
woe_test['RevolvingUtilizationOfUnsecuredLines'] = get_woe(train['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1)
woe_test['age'] = get_woe(train['age'], cutx2, woex2)
woe_test['NumberOfTime30-59DaysPastDueNotWorse'] = get_woe(train['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3)
woe_test['DebtRatio'] = get_woe(train['DebtRatio'], cutx4, woex4)
woe_test['MonthlyIncome'] = get_woe(train['MonthlyIncome'], cutx5, woex5)
woe_test['NumberOfOpenCreditLinesAndLoans'] = get_woe(train['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6)
woe_test['NumberOfTimes90DaysLate'] = get_woe(train['NumberOfTimes90DaysLate'], cutx7, woex7)
woe_test['NumberRealEstateLoansOrLines'] = get_woe(train['NumberRealEstateLoansOrLines'], cutx8, woex8)
woe_test['NumberOfTime60-89DaysPastDueNotWorse'] = get_woe(train['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9)
woe_test['NumberOfDependents'] = get_woe(train['NumberOfDependents'], cutx10, woex10)

import statsmodels.api as sm
from sklearn.metrics import roc_curve,auc

Y = woe_train['SeriousDlqin2yrs']
X = woe_train.drop(['SeriousDlqin2yrs','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'],axis = 1)
X1 = sm.add_constant(X)
logit = sm.Logit(Y,X1)
Logit_model = logit.fit()
print('输出拟合的各项系数')
print(Logit_model.params)

Y_test = woe_test['SeriousDlqin2yrs']
X_test = woe_test.drop(['SeriousDlqin2yrs','DebtRatio','MonthlyIncome',
                        'NumberOfOpenCreditLinesAndLoans',
                        'NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)
X3 = sm.add_constant(X_test)
resu = Logit_model.predict(X3)
fpr,tpr,threshold = roc_curve(Y_test,resu)
rocauc = auc(fpr,tpr)
plt.plot(fpr,tpr,'y',label='AUC=%0.2f' % rocauc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'p--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TurePositive')
plt.xlabel('FalsePositive')
plt.savefig('模型AUC曲线.png',dpi=300,bbox_inches='tight')
print('模型AUC曲线：')
plt.show()

#定义get_score函数用于计算各个分箱的基础得分
# Define get_score function to calculate the basic score for each bin
def get_score(coe, woe, factor):
    scores = []
    for w in woe:
        score = round(coe * w * factor, 0)
        scores.append(score)
    return scores

# Define compute_score function to calculate the basic score corresponding to each attribute value
def compute_score(feat, cut, score):
    res = []
    for index, value in feat.items():  # Use .items() instead of .iteritems()
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                break  # Correct loop exit condition
            else:
                j -= 1
                m -= 1
        res.append(score[m])
    return res


import math
import numpy as np

coe = Logit_model.params
p = 20 / math.log(2)
q = 600 - 20 * math.log(20) / math.log(2)
baseScore = round(q + p * coe[0], 0)

x1 = get_score(coe[1], woex1, p)
x2 = get_score(coe[2], woex2, p)
x3 = get_score(coe[3], woex3, p)
x7 = get_score(coe[4], woex7, p)
x9 = get_score(coe[5], woex9, p)

# Print the scores for the first feature
print('第1列属性取值在各分箱段对应的分数')
print(x1)

# Calculate scores for each test sample
test['BaseScore'] = np.zeros(len(test)) + baseScore
test['x1'] = compute_score(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1)
test['x2'] = compute_score(test['age'], cutx2, x2)
test['x3'] = compute_score(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3)
test['x7'] = compute_score(test['NumberOfTimes90DaysLate'], cutx7, x7)
test['x9'] = compute_score(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9)

test['Score'] = test['x1'] + test['x2'] + test['x3'] + test['x7'] + test['x9'] + baseScore


Normal = test.loc[test['SeriousDlqin2yrs']==1]
Charged = test.loc[test['SeriousDlqin2yrs']==0]

print('测试集中正常客户组信用评分统计描述')
print(Normal['Score'].describe())
print('测试集中违约客户组信用评分统计描述')
print(Charged['Score'].describe())

import seaborn as sns
plt.figure(figsize = (10,4))
sns.kdeplot(Normal['Score'],label = 'normal',linewidth = 2,linestyle = '--')
sns.kdeplot(Charged['Score'],label = 'charged',linewidth = 2,linestyle = '-')
plt.xlabel('Score',fontdict = {'size':10})
plt.ylabel('probability',fontdict = {'size':10})
plt.title('normal/charged',fontdict={'size':18})
plt.savefig('违约与正常客户的信用分布情况.png',dpi = 300,bbox_inches = 'tight')
plt.show()

#将训练好的模型用于客户信用评分
cusInfo = {'RevolvingUtilizationOfUnsecuredLines':0.509791452,'age':63,'NumberOfTime30-59DaysPastDueNotWorse':0,
           'NumberOfTime60-89DaysPastDueNotWorse':0,'DebtRatio':0.342429365,'MonthlyIncome':4140,
           'NumberOfOpenCreditLinesAndLoans':4,'NumberOfTimes90DaysLate':0,'NumberRealEstateLoansOrLines':0,
           'NumberOfTime60-89DaysPastDueNotWorse':0,'NumberOfDependents':1}
custData = pd.DataFrame(cusInfo,pd.Index(range((1))))
custData.drop(['DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'],axis = 1)

custData['x1'] = compute_score(custData['RevolvingUtilizationOfUnsecuredLines'], cutx1,x1)
custData['x2'] = compute_score(custData['age'], cutx2,x2)
custData['x3'] = compute_score(custData['NumberOfTime30-59DaysPastDueNotWorse'], cutx3,x3)
custData['x7'] = compute_score(custData['NumberOfTimes90DaysLate'], cutx7,x7)
custData['x9'] = compute_score(custData['NumberOfTime60-89DaysPastDueNotWorse'], cutx9,x9)

custData['Score'] = custData['x1']+custData['x2']+custData['x3']+custData['x7']+custData['x9']+baseScore
print('The credit score for the customer is：')
print(custData.loc[0,'Score'])








