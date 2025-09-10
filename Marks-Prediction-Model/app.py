from flask import Flask ,request , jsonify
import joblib

model=joblib.load("model.pkl")

app=Flask(__name__)

@app.route('/')
def home():
    return "Welcome to student marks prediction API!"

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.get_json()
        hours=data.get("hours")
        
        if hours is None:
            return jsonify({"error":"Missing 'hours' in request"}),400
        
        prediction=model.predict([[hours]])
        return jsonify({"hours":hours,"predicted_marks":round(prediction[0],2)})
    
    except Exception as e:
        return jsonify({"error":str(e)}),500
    
if __name__=='__main__':
    app.run(debug=True ,port=5000)
