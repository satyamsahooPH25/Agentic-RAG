from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from sec_downloader.types import RequestedFilings
from dotenv import load_dotenv
import os
from sec_api import PdfGeneratorApi
from sec_downloader import Downloader
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import uvicorn
import shutil

dl = Downloader("MyCompanyName", "email@example.com")


load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportRequest(BaseModel):
    company_tickers: list
    year: list


def save_to_google_drive(file_path, folder_id):
    """
    Uploads a file to a specified Google Drive folder.

    Args:
        file_path (str): The path to the file to be uploaded.
        folder_id (str): The ID of the Google Drive folder where the file will be uploaded.

    Returns:
        None

    Raises:
        google.auth.exceptions.GoogleAuthError: If there is an issue with the Google authentication.
        googleapiclient.errors.HttpError: If there is an issue with the Google Drive API request.

    Example:
        save_to_google_drive("path/to/your/file.pdf", "your_folder_id")
    """
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    SERVICE_ACCOUNT_FILE = "credentials.json"

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    service = build("drive", "v3", credentials=credentials)

    file_metadata = {"name": os.path.basename(file_path), "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="application/pdf")

    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )

    print(f"File ID: {file.get('id')}")


def upload_report(company_ticker: str, year: int):
    """
    Uploads the 10-K report for a given company and year to Google Drive.

    This function retrieves the metadata for the 10-K filings of the specified company,
    finds the filing for the specified year, generates a PDF of the filing, and uploads
    the PDF to Google Drive.

    Args:
        company_ticker (str): The ticker symbol of the company.
        year (int): The year of the 10-K report to be uploaded.

    Raises:
        ValueError: If no 10-K report is found for the specified year.
    """
    # Your upload_report function implementation here
    metadatas = dl.get_filing_metadatas(
        RequestedFilings(ticker_or_cik=company_ticker, form_type="10-K", limit=10)
    )
    link = ""
    for item in metadatas:
        if int(item.filing_date.split("-")[0]) == year:
            link = item.primary_doc_url
            break

    if not link:
        raise ValueError(f"No 10-K report found for {company_ticker} in {year}.")

    pdfGeneratorApi = PdfGeneratorApi(os.getenv("SEC_API"))

    # Form 10-K exhibit URL
    edgar_file_url = link

    pdf_file = pdfGeneratorApi.get_pdf(edgar_file_url)

    with open(f"{company_ticker}_{year}_10K.pdf", "wb") as f:
        f.write(pdf_file)

    # Save the file to Google Drive
    save_to_google_drive(
        f"{company_ticker}_{year}_10K.pdf", "1Yi0VqUIg4a1mmJxunXqbnQdlA_IF17Zx"
    )


def upload_multiple_report(company_tickers, year):
    """
    Uploads multiple reports for given company tickers and years.

    This function iterates over the provided company tickers and years,
    and uploads the corresponding reports by calling the `upload_report` function.

    Args:
        company_tickers (list): A list of company ticker symbols.
        year (list): A list of years corresponding to the company tickers.

    Returns:
        None
    """
    for ticker,yr in zip(company_tickers,year):
        upload_report(ticker,yr)


#this endpoint is for uploading the file from the storage

@app.post("/upload_file/")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload a financial document file to Google Drive.

    Args:
        file (UploadFile): The file uploaded by the user.

    Returns:
        dict: A success message and the file ID on Google Drive.
    """
    try:
        # Save the uploaded file locally
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Upload the file to Google Drive
        folder_id = "1Yi0VqUIg4a1mmJxunXqbnQdlA_IF17Zx"  # Replace with your Google Drive folder ID
        file_id = save_to_google_drive(temp_file_path, folder_id)

        # Clean up: Remove the temp file
        os.remove(temp_file_path)

        return {"message": "File uploaded successfully", "file_id": file_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")




#this is the endpoint for selecting the ticker and choosing the year just

@app.post("/upload_report/")
async def upload_report_endpoint(request: ReportRequest):
    try:
        upload_multiple_report(request.company_tickers, request.year)
        return {"message": "Report uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8156)