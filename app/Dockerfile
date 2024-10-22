# Use an official Python runtime as a parent image
FROM python:3.9-slim
RUN apt-get -y install git
# Set the working directory in the container
WORKDIR /app

COPY requirements.txt /app/.

RUN pip install --no-cache-dir -r requirements.txt
# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt


# Make port 8080 available to the world outside this container
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#CMD ["fastapi", "run", "main.py", "--port", "8000"]