
# 🛠️ Feature Engineering in EDA

Feature Engineering is the process of transforming raw data into meaningful features that better represent the underlying problem to the predictive models. It plays a crucial role in Exploratory Data Analysis (EDA) and Machine Learning.

---

## 📌 Why Feature Engineering?

- Improve model performance
- Handle missing data or inconsistencies
- Uncover hidden patterns
- Encode domain knowledge
- Reduce dimensionality

---

## 🔍 Types of Feature Engineering

### 1. **Creating New Features from Existing Ones**

#### 📌 Example: Extracting datetime features
```python
import pandas as pd

df = pd.DataFrame({
    'order_date': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-10'])
})

df['order_month'] = df['order_date'].dt.month
df['order_day'] = df['order_date'].dt.day
df['order_weekday'] = df['order_date'].dt.day_name()

print(df)
````

### 2. **Binning/Discretization**

#### 📌 Example: Age groups

```python
df['age'] = [15, 25, 35, 60]
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60], labels=['Teen', 'Adult', 'Senior'])
```

### 3. **Combining Features**

#### 📌 Example: Full name from first and last name

```python
df['full_name'] = df['first_name'] + " " + df['last_name']
```

### 4. **Text Feature Engineering**

#### 📌 Example: Word count and character count

```python
df['text'] = ['Hello world', 'Feature engineering is fun']
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['char_count'] = df['text'].apply(len)
```

### 5. **Mathematical Transformations**

#### 📌 Example: Ratio of two features

```python
df['price'] = [100, 200, 150]
df['quantity'] = [2, 4, 3]
df['price_per_unit'] = df['price'] / df['quantity']
```

---

## 💡 Best Practices

* Analyze correlation of engineered features with the target.
* Avoid overfitting with too many synthetic features.
* Use domain knowledge to create meaningful features.
* Visualize new features to verify distribution and usefulness.

---

## 📈 Tools & Libraries

* `pandas` – for manipulating data
* `numpy` – for numerical calculations
* `sklearn.preprocessing` – for scaling and encoding
* `feature-engine` – for advanced transformations

---

## 📚 References

* [Feature Engineering for Machine Learning (Book by Alice Zheng)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
* [Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)

---

## 📦 Sample Dataset

You can use datasets like:

* Titanic (from Seaborn or Kaggle)
* House Prices
* Sales or retail datasets

---

## ✅ Example Workflow Summary

```python
# Load dataset
df = pd.read_csv('data.csv')

# Create new time-based features
df['year'] = pd.to_datetime(df['date']).dt.year

# Combine columns
df['customer_info'] = df['name'] + ' - ' + df['location']

# Encode categorical features
df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})
```

---

> 🧠 “Good features allow models to learn patterns more effectively.”

