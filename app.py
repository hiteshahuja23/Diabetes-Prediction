from flask import Flask,render_template,render_template,request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            number_of_times_pregnant=float(request.form['pregnancies'])
            plasma_glucose_concentration = float(request.form['glucose'])
            diastolic_blood_pressure = float(request.form['bp'])
            triceps_skinfold_thickness = float(request.form['skin_thickness'])
            serum_insulin = float(request.form['insulin'])
            body_mass_index = float(request.form['bmi'])
            diabetes_pedigree_function = float(request.form['diabetespedigree'])
            age = request.form['age']

            # Loading the saved models into memory
            filename_scaler = 'final_scaler.sav'
            filename = 'final_log_reg.sav'
            scaler_model = pickle.load(open(filename_scaler, 'rb'))
            loaded_model = pickle.load(open(filename, 'rb'))

            # predictions using the loaded model file
            scaled_data = scaler_model.transform([[number_of_times_pregnant,plasma_glucose_concentration,diastolic_blood_pressure,triceps_skinfold_thickness,serum_insulin,body_mass_index,diabetes_pedigree_function, age]])
            prediction = loaded_model.predict(scaled_data)
            print('prediction is', prediction[0])
            if prediction[0]==1:
                result = 'The Patient is Diabetic. \n Please consult a doctor soon'
            else:
                result = 'The Patient is not Diabetic. \n Enjoy Life!'
            
            return render_template('results.html',result=result)
            # showing the prediction results in a UI
            #return jsonify(result)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')





if __name__ == "__main__":
    app.run(debug=True)
    