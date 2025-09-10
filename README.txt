========================
Depression Detection API
========================

This API serves your trained XLM-RoBERTa model for depression detection in Urdu sarcastic text.

How to Run:
-----------
1. Place your trained model file `xlmr_depression_classifier.pt` in the same folder as app.py.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the API:
   uvicorn app:app --reload
4. Test in browser:
   http://127.0.0.1:8000/docs
5. Example request (POST to /predict):
   {
     "text": "یہ جملہ طنزیہ ہے اور افسردگی ظاہر کرتا ہے"
   }

Response:
---------
{
  "input": "یہ جملہ طنزیہ ہے اور افسردگی ظاہر کرتا ہے",
  "prediction": "Depressed"
}
