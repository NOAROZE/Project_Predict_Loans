import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib


def loading_and_preprocess_the_dataset(file):
    """
        טוען את הנתונים מהקובץ ומבצע עיבוד מקדים בסיסי:
        טיפול בערכים חסרים והמרת משתנים קטגוריאליים למספרים.
    """
    # טעינת הנתונים
    df = pd.read_csv(file)

    # טיפול בערכים חסרים (null)
    # נמלא ערכים חסרים בעמודות מספריות בחציון (median) ובעמודות קטגוריאליות בשכיח (mode)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    # המרת עמודות טקסטואליות למספרים שהמודל יכול להבין
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    return df


def train_and_save_model(df, model_filename='loan_model.pkl'):
    """
        מאמן מודל SVC בתוך Pipeline הכולל נרמול נתונים (StandardScaler),
        ושומר את המודל המאומן לקובץ.
    """
    # הגדרת הפיצ'רים (X) ותווית המטרה (y)
    X = df[['Married', 'Education', 'ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
    y =df['Loan_Status']

    # חלוקה לנתוני אימון ונתוני בדיקה (כדי לחשב דיוק אמיתי)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # יצירת ה-Pipeline: קודם מנרמלים את הנתונים, ואז מעבירים למודל הסיווג
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])

    # חישוב התחזיות על נתוני ה-train
    pipeline.fit(X_train, y_train)

    # חישוב התחזיות על נתוני ה-test
    y_pred = pipeline.predict(X_test)

    # חישוב רמת הדיוק
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    model_data = {
        'model': pipeline,
        'accuracy': accuracy
    }

    # שמירת המודל המאומן לקובץ
    joblib.dump(model_data, model_filename)
    print(f"The model has been trained and saved to a file: {model_filename}")


if __name__ == "__main__":
    # הנתיב לקובץ הנתונים
    data_csv = 'dataset_loans.csv'

    # טוען ומעבד את הנתונים
    print('Loading and processing data')
    processed_data = loading_and_preprocess_the_dataset(data_csv)

    # אימון ושמירת המודל
    print('Trains and maintains the model')
    train_and_save_model(processed_data)