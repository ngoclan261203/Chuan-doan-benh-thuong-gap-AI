import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk, messagebox



######## Đọc dữ liệu
sym_des = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\symtoms_df_trans.csv")
precautions = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\precautions_df.csv")
description = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\description.csv")
medications = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\medications.csv")
diets = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\diets.csv")
workout = pd.read_csv(r"C:\Users\NGOC LAN\Desktop\AI - Copy - Copy\workout_df.csv")

# sym_des: Liệt kê các bệnh và triệu chứng liên quan.
# precautions: Biện pháp phòng ngừa cho từng bệnh.
# description: Mô tả chi tiết của từng bệnh.
# medications: Thuốc điều trị bệnh.
# diets: Chế độ ăn uống phù hợp.
# workout: Các hoạt động thể chất được khuyến nghị.



########### Chuẩn bị dữ liệu

##xử lý dữ liệu triệu chứng
all_symptoms = sym_des.iloc[:, 1:].values.flatten() 
#Trích xuất toàn bộ các triệu chứng từ bảng dữ liệu sym_des.
#Sử dụng .iloc[] để chọn dữ liệu bằng chỉ số hàng, cột.
# : (hàng): Lấy toàn bộ các hàng.
# 1: (cột): Lấy tất cả các cột từ cột thứ 2 trở đi (bỏ qua cột đầu tiên, vì nó là cột Disease).
# .values:Trả về dữ liệu dưới dạng mảng NumPy thay vì DataFrame.
# .flatten():Biến đổi mảng hai chiều (mỗi hàng chứa các triệu chứng của một bệnh) thành mảng một chiều (danh sách tất cả các triệu chứng).
all_symptoms = [symptom for symptom in all_symptoms if pd.notna(symptom)]
#oại bỏ tất cả các giá trị rỗng (NaN) trong danh sách triệu chứng.
all_symptoms = sorted(set(all_symptoms))
#Tạo danh sách các triệu chứng duy nhất loại bỏ các triệu chứng trùng lặp
#sắp xếp theo thứ tự bảng chữ cái.
symptoms_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
#Tạo một dictionary ánh xạ giữa triệu chứng và chỉ số duy nhất (ID).
#enumerate(all_symptoms):Trả về các cặp (index, symptom) khi duyệt qua danh sách all_symptoms.
#vd:[(0, 'Fever'), (1, 'Headache'), (2, 'Cough')]
#kết qảu: Tạo một dictionary, trong đó: Key(Là triệu chứng (symptom).);Value: Là chỉ số tương ứng (idx).
#vd {'Fever': 0,'Headache': 1,'Cough': 2}

##xử lý dữ liệu tên bệnh
unique_diseases = sym_des['Disease'].unique()
#Lấy cột Disease từ DataFrame sym_des.
# unique: trả về mảng chứa các gt duy nhất k trùng lặp
diseases_list = {idx: disease for idx, disease in enumerate(unique_diseases)}
# Tạo một dictionary ánh xạ giữa chỉ số duy nhất và tên bệnh.
# Trả về các cặp (index, disease) khi duyệt qua danh sách unique_diseases.
# kq: Tạo một dictionary, trong đó:Key: Là chỉ số (idx); Value: Là tên bệnh (disease).
# {0: 'Flu',1: 'Cold',2: 'Covid'}



############Xây dựng tập dữ liệu huấn luyện

##Khởi tạo các mảng X và y
X = np.zeros((len(sym_des), len(symptoms_dict)))
# X: Ma trận đặc trưng (features).
#số hàng (len(sym_des)): Tương ứng với số bệnh trong dữ liệu (mỗi hàng đại diện cho một bệnh).
#Số cột (len(symptoms_dict)): Tương ứng với số triệu chứng duy nhất (mỗi cột đại diện cho một triệu chứng).
# Mỗi phần tử trong ma trận X[i, j] sẽ được gán giá trị:
# 1: Nếu bệnh tại hàng i có triệu chứng tương ứng với cột j.
# 0: Nếu không có triệu chứng đó./
y = np.zeros(len(sym_des))
# y: Mảng mục tiêu (labels).
# Số phần tử (len(sym_des)): Tương ứng với số bệnh.
# Mỗi phần tử y[i] lưu trữ chỉ số (ID) của bệnh ở hàng i.

for i, row in sym_des.iterrows():
    #sym_des.iterrows():Duyệt qua từng hàng (bệnh) trong DataFrame sym_des.
    # row: Đại diện cho một hàng trong DataFrame, chứa:
    # row['Disease']: Tên bệnh trong cột đầu tiên.
    # row[1:]: Danh sách triệu chứng (các cột từ thứ hai trở đi).
    # i: Là chỉ số (index) của hàng trong DataFrame.
    for symptom in row[1:]:##Xử lý các triệu chứng của một bệnh
        # Duyệt qua từng triệu chứng của bệnh tại hàng i.
        if pd.notna(symptom):
            # Kiểm tra xem triệu chứng hiện tại có phải giá trị hợp lệ không (loại bỏ giá trị NaN).
            X[i, symptoms_dict[symptom]] = 1
            # symptoms_dict[symptom]]:Trả về chỉ số (index) của triệu chứng trong danh sách các triệu chứng duy nhất (all_symptoms)
            # tại hàng i cột (chỉ số index triệu trứng) lưu gt =1
    y[i] = list(diseases_list.values()).index(row['Disease'])
    #Cập nhật mảng mục tiêu y với chỉ số (ID) của bệnh hiện tại.
    #có nghĩa là sẽ lưu gt id của tên bệnh tại dòng ddang xét(id lấy từ diseases_list )
    #row['Disease']:Lấy tên bệnh từ cột đầu tiên của hàng hiện tại.
    # list(diseases_list.values()):Chuyển giá trị của dictionary diseases_list (danh sách các bệnh) thành một danh sách.
    # Ví dụ: Nếu diseases_list = {0: 'Flu', 1: 'Cold', 2: 'Covid'}, thì:
    # list(diseases_list.values()) = ['Flu', 'Cold', 'Covid'].
    # .index(row['Disease']):Lấy chỉ số (index) của tên bệnh trong danh sách diseases_list.values().



# Huấn luyện mô hình SVM cho hệ thống gợi ý
svc = SVC() #Tạo một mô hình SVM phân loại.
svc.fit(X, y)#Huấn luyện mô hình trên dữ liệu X (triệu chứng) và y (bệnh).




# **Thêm phần đánh giá mô hình vào đây**
from sklearn.metrics import accuracy_score, classification_report

# Tạo tập dữ liệu kiểm tra (train-test split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện lại mô hình với dữ liệu huấn luyện
svc.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = svc.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# In báo cáo chi tiết các chỉ số đánh giá
print(classification_report(y_test, y_pred))

# Đánh giá với Cross-validation (nếu không có test set)
cross_val_scores = cross_val_score(svc, X, y, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy scores: {cross_val_scores}')
print(f'Mean cross-validation accuracy: {cross_val_scores.mean():.2f}')



###### Hàm dự đoán
#Hàm nhận ds các triệu chứng của một bệnh nhân (patient_symptoms) 
# và trả về dự đoán bệnh mà bệnh nhân đó có thể mắc phải.
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    #Tạo một mảng NumPy (input_vector) với số phần tử bằng số triệu chứng duy nhất trong symptoms_dict.
    #ạo một mảng ban đầu toàn là gt 0, có độ dài bằng số triệu chứng.
    for item in patient_symptoms:#Duyệt qua từng triệu chứng trong danh sách triệu chứng đầu vào
        if item in symptoms_dict:#nếu triệu chứng có trong ds triệu chứng
            input_vector[symptoms_dict[item]] = 1
            #Gắn giá trị 1 vào vị trí tương ứng trong mảng input_vector.
    #hết vòng for sẽ trả về 1 vecto có các gt 0,1
    # mục đích của vòng for là Gắn giá trị 1 cho các triệu chứng có trong danh sách đầu vào

    # Dự đoán chỉ số của bệnh
    predicted_index = svc.predict([input_vector])[0]
    #svc.predict():Hàm dự đoán của mô hình SVM (svc).
    # Đầu vào: [input_vector] (đóng gói thành danh sách vì mô hình yêu cầu đầu vào là 2D).
    # Đầu ra: chỉ số (ID) của bệnh dự đoán.
    return diseases_list[predicted_index]
    #Trả về tên bệnh dự đoán




#### Hàm lấy thông tin chi tiết
def helper(dis):
    #dis: tên bệnh(đầu vào)
    desc = description[description['Disease'] == dis]['Description']
    #description['Disease'] == dis:Lọc các hàng trong DataFrame description nơi cột Disease có giá trị bằng tên bệnh dis.
    # cả dòng trên sẽ lấy gt trong cột 'Description' tương ứng sau kih lọc
    desc = " ".join(desc.values)
    #desc.values:Trả về giá trị của cột dưới dạng mảng NumPy.
    # vd :desc.values = ["Flu is a viral infection"]
    #" ".join(desc.values):nối tất cả các pt trg mảng desc.values thành 1 chuỗi duy nhất, cách nhua bằng dấu cách

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    #Lấy danh sách các biện pháp phòng ngừa của bệnh từ bảng precautions.
    pre = pre.values.tolist()#chuyển thành ds

    med = medications[medications['Disease'] == dis]['Medication']
    med = [str(item).replace("'", "").replace("[", "").replace("]", "") for item in med]  # Loại bỏ dấu nháy đơn và ngoặc vuông
    #Xử lý chuỗi thuốc để loại bỏ ký tự ', [, ] nếu chúng xuất hiện.
    #Điều này thường được thực hiện để làm sạch dữ liệu nếu cột Medication chứa danh sách hoặc chuỗi không đúng định dạng.
    
    die = diets[diets['Disease'] == dis]['Diet']
    die = [str(item).replace("'", "").replace("[", "").replace("]", "") for item in die]  # Loại bỏ dấu nháy đơn và ngoặc vuông

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = wrkout.values.tolist()

    return desc, pre, med, die, wrkout




######### Hàm dự đoán bệnh khi nhấn nút chuẩn đoán
def predict_disease():
    user_input = symptoms_entry.get().lower()  
    # Chuyển toàn bộ chữ nhập vào thành chữ thường
    #symptoms_entry.get(): lấy gt của ô nhập triệu chứng
    if not user_input.strip():
    #user_input.strip():Loại bỏ các khoảng trắng ở đầu và cuối chuỗi: Kiểm tra xem người dùng có thực sự nhập dữ liệu hay không.
    #Nếu chuỗi trống sau khi xóa khoảng trắng, hiện hộp thoại cảnh báo
    #và dừng không tiếp tục xuwrl ý 
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập triệu chứng.")
        return

    # Tách các triệu chứng, chuyển thành chữ thường và loại bỏ khoảng trắng thừa
    user_symptoms = [symptom.strip() for symptom in user_input.split(',') if symptom.strip()]
    #tách chuỗi khi gặp dấu phẩy sau đó xóa khoảng trắng xung quanh các pt vừa đc tách
    #if symptom.strip(): Bỏ qua các phần tử trống hoặc chỉ chứa khoảng trắng.
    # kq trả về là 1 mảng các triệu chứng viết thg và đã đc xử lý
    valid_symptoms = [sym for sym in user_symptoms if sym in symptoms_dict]
    #Kiểm tra tính hợp lệ của các triệu chứng
    #Lọc chỉ giữ lại những triệu chứng hợp lệ từ danh sách user_symptoms.

    #Xử lý khi không có triệu chứng hợp lệ(rỗng)
    if not valid_symptoms:
        messagebox.showerror("Lỗi", "Không có triệu chứng hợp lệ. Vui lòng nhập lại.")
        return

    # Dự đoán và hiển thị kết quả
    predicted_disease = get_predicted_value(valid_symptoms)#gọi hàm dụ đoán bệnh bên trên
    result_label.config(text=f"{predicted_disease}")

    global current_disease
    current_disease = predicted_disease
    #Biến toàn cục current_disease lưu tên bệnh hiện tại để sử dụng cho các tác vụ khác trong chương trình.
    # mỗi lần gọi hàm này bến toàn cục sẽ đc khai báo và gắn lại gt

    # Xóa nội dung của Label kết quả chi tiết
    details_label.config(text="")  # Đặt lại nội dung trống




######## hiện thị mô tả khi nhấn nút mô tả
def show_description():
    if 'current_disease' not in globals():
    # Kiểm tra xem có bệnh nào đã được dự đoán hay chưa
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return
        #nếu chưa thì dùng k xử lý 

    desc, _, _, _, _ = helper(current_disease)
    #Lấy mô tả bệnh từ hàm helper()
    
    # Xóa nội dung cũ trước khi hiển thị mô tả mới
    details_label.config(text="")  # Đặt lại nội dung trống
    
    # Cập nhật nội dung mới
    description_title = "Mô tả bệnh:"+"\n"
    description_content = desc
    details_label.config(text=f"{description_title}\n{description_content}")



######## hiện thị phòng ngừa khi nhấn nút phnogf ngừa
def show_precautions():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return

    _, pre, _, _, _ = helper(current_disease)
    
    # Xóa nội dung cũ trước khi hiển thị biện pháp mới
    details_label.config(text="")  # Đặt lại nội dung trống
    
    # Cập nhật nội dung mới
    precautions_title = "Biện pháp phòng ngừa bệnh:"+"\n"
    precautions_content = "\n".join(pre[0]) if pre else "Không có thông tin"
    details_label.config(text=f"{precautions_title}\n{precautions_content}")


def show_medications():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return

    _, _, med, _, _ = helper(current_disease)
    
    # Xóa nội dung cũ trước khi hiển thị thuốc chữa trị mới
    details_label.config(text="")  # Đặt lại nội dung trống
    
    # Cập nhật nội dung mới
    medications_title = "Tham khảo thuốc chữa trị:"+"\n"
    medications_content = "\n".join(med) if med else "Không có thông tin"
    details_label.config(text=f"{medications_title}\n{medications_content}")

def show_diet():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return

    _, _, _, die, _ = helper(current_disease)
    
    # Xóa nội dung cũ trước khi hiển thị chế độ ăn tham khảo mới
    details_label.config(text="")  # Đặt lại nội dung trống
    
    # Cập nhật nội dung mới
    diet_title = "Chế độ ăn tham khảo:"+"\n"
    diet_content = "\n".join(die) if die else "Không có thông tin"
    details_label.config(text=f"{diet_title}\n{diet_content}")

def show_workout():
    if 'current_disease' not in globals():
        messagebox.showwarning("Cảnh báo", "Chưa có bệnh dự đoán. Vui lòng dự đoán bệnh trước.")
        return

    _, _, _, _, wrkout = helper(current_disease)
    
    # Xóa nội dung cũ trước khi hiển thị hoạt động mới
    details_label.config(text="")  # Đặt lại nội dung trống
    
    # Cập nhật nội dung mới
    workout_title = "Các hoạt động nên thực hiện:"+"\n"
    workout_content = "\n".join(wrkout) if wrkout else "Không có thông tin"
    details_label.config(text=f"{workout_title}\n{workout_content}")





# Giao diện chính
root = tk.Tk()
root.title("Hệ thống dự đoán bệnh")
root.geometry("1000x700")
root.configure(bg="#f9f9f9")

# Tiêu đề chính
title_label = tk.Label(root, text="Chuẩn đoán bệnh thường gặp", font=("Arial", 22, "bold"), bg="white", fg="black")
title_label.pack(pady=10)

# Khung nhập triệu chứng
input_frame = tk.Frame(root, bg="black", padx=15, pady=10)
input_frame.pack(pady=10, fill="x", padx=20)

tk.Label(input_frame, text="Nhập triệu chứng của bạn (cách nhau dấu phẩy):", font=("Arial", 14), bg="black", fg="white").pack(side="left", padx=10)
symptoms_entry = tk.Entry(input_frame, font=("Arial", 14), width=50)
symptoms_entry.pack(side="left", padx=10)
tk.Button(input_frame, text="Dự đoán bệnh", font=("Arial", 14), bg="red", fg="white", command=predict_disease).pack(side="right", padx=10)

# Khung bệnh dự đoán
prediction_frame = tk.Frame(root, bg="white", padx=10, pady=10, relief="solid", borderwidth=1)
prediction_frame.pack(pady=10, fill="x", padx=20)

tk.Label(prediction_frame, text="Kết quả dự đoán:", font=("Arial", 14, "bold"), bg="white", fg="black").grid(row=0, column=0, sticky="w", padx=10, pady=5)
result_label = tk.Label(prediction_frame, text="(Hiển thị tại đây)", font=("Arial", 14), bg="white", fg="black", wraplength=800, justify="left")
result_label.grid(row=0, column=1, sticky="w", padx=10, pady=5)


# Khung chi tiết kết quả
details_frame = tk.Frame(root, bg="white", padx=10, pady=10, relief="solid", borderwidth=1)
details_frame.pack(pady=10, fill="both", expand=True, padx=20)

details_label = tk.Label(details_frame, text="", font=("Arial", 12), bg="white", fg="black", wraplength=900, justify="left", anchor="nw")
details_label.pack(fill="both", expand=True, padx=10, pady=10)

# Các nút để hiển thị chi tiết
buttons_frame = tk.Frame(root, bg="white", pady=10)
buttons_frame.pack()

tk.Button(buttons_frame, text="Mô tả", font=("Arial", 14), bg="blue", fg="white", width=12, command=show_description).pack(side="left", padx=10, pady=10)
tk.Button(buttons_frame, text="Biện pháp", font=("Arial", 14), bg="pink", fg="white", width=12, command=show_precautions).pack(side="left", padx=10, pady=10)
tk.Button(buttons_frame, text="Thuốc chữa", font=("Arial", 14), bg="red", fg="white", width=12, command=show_medications).pack(side="left", padx=10, pady=10)
tk.Button(buttons_frame, text="Chế độ ăn", font=("Arial", 14), bg="#ffa500", fg="white", width=12, command=show_diet).pack(side="left", padx=10, pady=10)
tk.Button(buttons_frame, text="Hoạt động", font=("Arial", 14), bg="green", fg="white", width=12, command=show_workout).pack(side="left", padx=10, pady=10)

root.mainloop()
