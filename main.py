import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk, messagebox

sym_des = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\symtoms_df_trans.csv")
precautions = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\precautions_df.csv")
description = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\description.csv")
medications = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\medications.csv")
diets = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\diets.csv")
workout = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\workout_df.csv")

all_symptoms = sym_des.iloc[:, 1:].values.flatten() 
all_symptoms = [symptom for symptom in all_symptoms if pd.notna(symptom)]
all_symptoms = sorted(set(all_symptoms))
symptoms_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

unique_diseases = sym_des['Disease'].unique()
diseases_list = {idx: disease for idx, disease in enumerate(unique_diseases)}

X = np.zeros((len(sym_des), len(symptoms_dict)))
y = np.zeros(len(sym_des))

for i, row in sym_des.iterrows():
    for symptom in row[1:]:
        if pd.notna(symptom):
            X[i, symptoms_dict[symptom]] = 1
    y[i] = list(diseases_list.values()).index(row['Disease'])

svc = SVC()
svc.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))

cross_val_scores = cross_val_score(svc, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy scores: {cross_val_scores}')
print(f'Mean cross-validation accuracy: {cross_val_scores.mean():.2f}')

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    predicted_index = svc.predict([input_vector])[0]
    return diseases_list[predicted_index]

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc.values)
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.tolist()
    med = medications[medications['Disease'] == dis]['Medication']
    med = [str(item).replace("'", "").replace("[", "").replace("]", "") for item in med]
    die = diets[diets['Disease'] == dis]['Diet']
    die = [str(item).replace("'", "").replace("[", "").replace("]", "") for item in die]
    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.values.tolist()
    return desc, pre, med, die, wrkout

def predict_disease():
    user_input = symptoms_entry.get().lower()
    if not user_input.strip():
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập triệu chứng.")
        return
    user_symptoms = [symptom.strip() for symptom in user_input.split(',') if symptom.strip()]
    valid_symptoms = [sym for sym in user_symptoms if sym in symptoms_dict]
    if not valid_symptoms:
        messagebox.showerror("Lỗi", "Không có triệu chứng hợp lệ. Vui lòng nhập lại.")
        return
    predicted_disease = get_predicted_value(valid_symptoms)
    result_label.config(text=f"{predicted_disease}")
    global current_disease
    current_disease = predicted_disease
    details_label.config(text="")

def show_description():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return
    desc, _, _, _, _ = helper(current_disease)
    details_label.config(text="")
    description_title = "Mô tả bệnh:" + "\n"
    description_content = desc
    details_label.config(text=f"{description_title}\n{description_content}")

def show_precautions():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return
    _, pre, _, _, _ = helper(current_disease)
    details_label.config(text="")
    precautions_title = "Biện pháp phòng ngừa bệnh:" + "\n"
    precautions_content = "\n".join(pre[0]) if pre else "Không có thông tin"
    details_label.config(text=f"{precautions_title}\n{precautions_content}")

def show_medications():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return
    _, _, med, _, _ = helper(current_disease)
    details_label.config(text="")
    medications_title = "Tham khảo thuốc chữa trị:" + "\n"
    medications_content = "\n".join(med) if med else "Không có thông tin"
    details_label.config(text=f"{medications_title}\n{medications_content}")

def show_diet():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return
    _, _, _, die, _ = helper(current_disease)
    details_label.config(text="")
    diet_title = "Chế độ ăn tham khảo:" + "\n"
    diet_content = "\n".join(die) if die else "Không có thông tin"
    details_label.config(text=f"{diet_title}\n{diet_content}")

def show_workout():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return
    _, _, _, _, wrkout = helper(current_disease)
    details_label.config(text="")
    workout_title = "Các hoạt động nên thực hiện:" + "\n"
    workout_content = "\n".join(wrkout) if wrkout else "Không có thông tin"
    details_label.config(text=f"{workout_title}\n{workout_content}")

root = tk.Tk()
root.title("Hệ thống dự đoán bệnh")
root.geometry("1000x700")
root.configure(bg="#f9f9f9")

title_label = tk.Label(root, text="Chuẩn đoán bệnh thường gặp", font=("Arial", 22, "bold"), bg="white", fg="black")
title_label.pack(pady=10)

input_frame = tk.Frame(root, bg="black", padx=15, pady=10)
input_frame.pack(pady=10, fill="x", padx=20)

tk.Label(input_frame, text="Nhập triệu chứng của bạn (cách nhau dấu phẩy):", font=("Arial", 14), bg="black", fg="white").pack(side="left", padx=10)
symptoms_entry = tk.Entry(input_frame, font=("Arial", 14), width=50)
symptoms_entry.pack(side="left", padx=10)
tk.Button(input_frame, text="Dự đoán bệnh", font=("Arial", 14), bg="red", fg="white", command=predict_disease).pack(side="right", padx=10)

prediction_frame = tk.Frame(root, bg="white", padx=10, pady=10, relief="solid", borderwidth=1)
prediction_frame.pack(pady=10, fill="x", padx=20)

tk.Label(prediction_frame, text="Kết quả dự đoán:", font=("Arial", 14, "bold"), bg="white", fg="black").grid(row=0, column=0, sticky="w", padx=10, pady=5)
result_label = tk.Label(prediction_frame, text="(Hiển thị tại đây)", font=("Arial", 14), bg="white", fg="black", wraplength=800, justify="left")
result_label.grid(row=0, column=1, sticky="w", padx=10, pady=5)

details_frame = tk.Frame(root, bg="white", padx=10, pady=10, relief="solid", borderwidth=1)
details_frame.pack(pady=10, fill="both", expand=True, padx=20)

details_label = tk.Label(details_frame, text="", font=("Arial", 12), bg="white", fg="black", wraplength=900, justify="left")
details_label.pack()

options_frame = tk.Frame(root, bg="white", padx=10, pady=10)
options_frame.pack(pady=10)

ttk.Button(options_frame, text="Xem mô tả bệnh", command=show_description).grid(row=0, column=0, padx=5, pady=5)
ttk.Button(options_frame, text="Biện pháp phòng ngừa", command=show_precautions).grid(row=0, column=1, padx=5, pady=5)
ttk.Button(options_frame, text="Thuốc tham khảo", command=show_medications).grid(row=0, column=2, padx=5, pady=5)
ttk.Button(options_frame, text="Chế độ ăn", command=show_diet).grid(row=0, column=3, padx=5, pady=5)
ttk.Button(options_frame, text="Hoạt động tham khảo", command=show_workout).grid(row=0, column=4, padx=5, pady=5)

root.mainloop()
