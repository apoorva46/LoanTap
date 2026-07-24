import pandas as pd

df = pd.read_csv("data/LoanTap.csv")

print("Purpose:")
print(df["purpose"].unique())

print("\nHome Ownership:")
print(df["home_ownership"].unique())

print("\nVerification Status:")
print(df["verification_status"].unique())

print("\nApplication Type:")
print(df["application_type"].unique())

print("\nGrade:")
print(df["grade"].unique())

print("\nSub Grade:")
print(df["sub_grade"].unique())

print("\nEmployment Length:")
print(df["emp_length"].unique())