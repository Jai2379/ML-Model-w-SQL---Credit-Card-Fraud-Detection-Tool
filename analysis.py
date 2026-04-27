import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

df = pd.read_csv("creditcard.csv")

df.head()


'''
plt.figure()

sns.histplot(df[df['Class']==0]['Amount'], label='Normal', kde=True)
sns.histplot(df[df['Class']==1]['Amount'], label='Fraud', kde=True)

plt.legend()
plt.title("Transaction Amount Distribution")

plt.show()

# X = All columns EXCEPT 'Class' (the features)
# y = ONLY the 'Class' column (the answer key)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

from sklearn.ensemble import RandomForestClassifier

# 1. Initialize the model
# n_estimators=100 means we are using 100 "trees" in our forest
# max_depth=5 prevents the trees from getting too complex (memorizing)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# 2. The Training Phase
# We give the AI the training data (X_train) and the answers (y_train)
print("The AI is now studying the patterns... this might take 20 seconds...")
model.fit(X_train, y_train)

print("Training Complete!")

#at this point, the model is ready to implement but we needa test it first to see how well it adapted to the data
# relate y_pred to y_test to see how well the model is doing

# 3. The Testing Phase
# The AI makes its guesses for the 20% of data it has NEVER seen
y_pred = model.predict(X_test)

# Let's look at the first 10 guesses
print("First 10 predictions (0 = Normal, 1 = Fraud):")
print(y_pred[:10])

# Compare the actual answers (y_test) to the AI's guesses (y_pred)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(y_test, y_pred))

#our data seems to be only somewhat accurate. It has a 68% accuracy score (not good).
#Next thing to look out for - "Which specific columns were you looking at the most when you caught those 68 people?"

# Pull the importance scores from the model
importances = pd.Series(model.feature_importances_, index=X.columns)

# Sort them and plot the top 10
importances.nlargest(10).plot(kind='barh')
plt.title("The AI's Top 10 Red Flags")
plt.show()'''

#--------Rewritten/Improved code---------------


# This creates a file named 'fraud_detection.db' in your folder
conn = sqlite3.connect('fraud_detection.db') 

# This takes your 'df' and turns it into a SQL table called 'transactions'
df.to_sql('transactions', conn, if_exists='replace', index=False)

print("Database Created! You now have a SQL database file in your folder.")

# Write your SQL query as a string
# this is to pull the specifics from ur 10 most important features
my_sql_query = """
SELECT V17, V14, V12, V10, V4, Amount, Class 
FROM transactions 
WHERE Amount > 0
"""

# Pull the data back into a new dataframe
sql_driven_df = pd.read_sql(my_sql_query, conn)

print("\nHere are the first 5 rows of the new dataframe created from your SQL query:")
print(sql_driven_df.head())

# --- STEP 14: THE SUPERCHARGED MODEL ---

# 1. Prepare Features and Target from your SQL dataframe
X_sql = sql_driven_df.drop('Class', axis=1)
y_sql = sql_driven_df['Class']

# 2. Apply SMOTE to balance the data
# This creates synthetic fraud cases so the AI has a 50/50 split to study
sm = SMOTE(random_state=42) #random_state = 42 is just a common choice to ensure reproducibility
X_res, y_res = sm.fit_resample(X_sql, y_sql)

'''print(f"\nBalanced dataset size: {len(X_res)} rows")
print(f"Fraud cases after SMOTE: {sum(y_res == 1)}")
'''
# 3. Split the BALANCED data
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# 4. Train the Final Model
# We use the same Random Forest, but now it has "better" data
final_model = RandomForestClassifier(n_estimators=100, 
                                     random_state=42, 
                                     max_depth = 10,
                                     min_samples_leaf=10,
                                     n_jobs=-1) #n_jobs=-1 uses all your CPU cores to speed up training
final_model.fit(X_train_res, y_train_res)

# 5. The Final Exam
y_pred_final = final_model.predict(X_test_res)

# Compare the actual answers (y_test) to the AI's guesses (y_pred)
cm = confusion_matrix(y_test_res, y_pred_final)

'''plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()'''


print("\n--- FINAL BALANCED MODEL REPORT ---")
print(classification_report(y_test_res, y_pred_final))

# This saves your tuned model so you can "load" it later without retraining
joblib.dump(final_model, 'final_fraud_detector.pkl')

print("Model saved! You can now deploy this to a website or server.")