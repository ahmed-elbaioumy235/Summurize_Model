from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from pptx_helper.helper import extract_text_from_pptx , clean_text 
from pptx_helper.summary import summary

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello there"

@app.post("/predict_pptx")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded audio file
    doc_data = await file.read()

    doc_path = "temp" + file.filename
    with open(doc_path, "wb") as f:
        f.write(doc_data)
    

    extracted_text = extract_text_from_pptx(doc_path)
    cleaned_text  = clean_text(extracted_text)

    data_summary = {
            "inputs": cleaned_text,
           }

    
    
    text_summary = summary(data_summary)
   


    summurization = text_summary[0]["summary_text"]

    os.remove(doc_path)

    return {
            "summurization" : summurization,
            }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)