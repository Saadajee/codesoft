#Base
FROM python:3.11-slim


#Working dir
WORKDIR /app

#Copy requirements First 
COPY requirements.txt .

#Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy project files
COPY . .

#Expose port for API
EXPOSE 8000


#Run FastAPI using uvicorn
CMD ["uvicorn", "app.main", "--host", "0.0.0.0", "--port", "8000"]