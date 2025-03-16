# base image
FROM python:3.12

# set working directory
WORKDIR /code

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download and cache the pretrained model
RUN python -c "import torch; import torchvision.models as models; models.densenet121(pretrained=True)"

# copy project
COPY ./app /code/app
COPY ./shared /code/shared

# expose port
EXPOSE 80

# run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

