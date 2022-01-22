# **Project Description**

## **Name: Mortgage Based Payment Ability** <br><br>

### <b> Description of the Project </b><br><br>

This project specifically focuses on finding an alternative method to predict mortgage prepayment risk of residential mortgage loans by using different machine learning techniques.

We have used: Freddie Mac’s Single Family Loan- Level Dataset

This dataset originally contains 95 features divided as Original data and Performance data.

In the dataset we make the use of 27datacolumns.

**The description of the dataset can be found in the following document:


 http://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf**

### <b> Description of the first Model </b><br><br>

We used **K-nearest neighbors Algorithm** as the second algorithm to train our dataset.

After doing the necessary **EDA** and **Data Preprocessing** we splited our model into train test sets 

```python

X_train,X_valid,y_train,y_valid = train_test_split(X_encoded,y,test_size= 0.2,random_state=0)

```
After that we scaled the training and validation data using the **StandardScaler**.

### **Balancing the imbalanced data and performing the training**<br>

We used the cocept of random oversampling to increase the number of minority class. Random oversampling involves randomly selecting examples from the minority class, with replacement, and adding them to the training dataset. 


``` python
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

os = RandomOverSampler(0.5)
X_train_ns,y_train_ns = os.fit_resample(x_train_std,y_train)
print("The number of classes before fit{}".format(Counter(y_train)))
print("The number of classes after fit{}".format(Counter(y_train_ns)))

```

### Took sample to decrease runtime.
Again splited the train dataset into Sample(20%) and Remaining(80%)

#split train data to take sample.
```python

X_train_rem, X_train_sample, y_train_rem, y_train_sample = train_test_split(X_train_ns, y_train_ns, test_size=0.2, random_state=42) 


```
After we took the sample, fit the **KNeighborsClassifier** 

```python

neighbors= np.arange(1,5)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

#loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sample, y_train_sample)
    
    # compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train_sample, y_train_sample)
    test_accuracy[i] = knn.score(X_valid_std, y_valid)
```


### **Model Analysis:**

we used confusion matrix and classification report to analyze the performance of our model.

**Result of confusion matrix:**


In case of confusion metric we can see that our model is correctly able to classify **81752(True positive**) and **1608(True Negative)** Value. But we have pretty decent number of **False Negative(953)** and **False positive(2048)** falsely classified by our model. **In our case we need to try to reduce the number of false positive because classifying a non-prepaid value as prepaid can result in more loss to the bank** than the otherway analogy. Also our data is highly imbalanced that could also be one reason why our model is performing that way. So we will also try balancing out the data set to see if will get some improvement.

From any company's point of view predicting prepaid when it is actually not is kind of more dangerous and we can see that our model has falsely classified a lot of unprepaid value as prepaid. So we also need to try to **reduce the false positive**.


**Result of classification Report**


From the report it's clear that the value of precision, recall and  f1 score is quite low for the class label 0 whereas for class label 1 it's pretty high.




**Result:**

```
The number of classes before fitCounter({True: 330873, False: 14569})
The number of classes after fitCounter({True: 330873, False: 165436})

```

Then we applied the similar model training approach like above and perform everything from the beginning.


### **Performance Analysis of the model after balancing the dataset**

We also used the technique of SMOTE in order to perform the data balance. But the performance of this technique was not as good as random oversampling.

Random Oversamping was better than SMOTE technique because:

* It's computational speed was exceptionally good than SMOTE.
* We were also able to get good recall and were able to decrease the false positive and has a good f1-score using **random over sampler** but in case of **SMOTE** the result for f1 score is worst.










