# base image
FROM python:3.12

# set working directory
WORKDIR /code

# install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# copy project
COPY ./app /code/app
COPY ./shared /code/shared

# Add User
RUN useradd -m myuser
USER myuser

# Download and cache the pretrained model
RUN python -c "import torch; import torchvision.models as models; models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)"

# expose port
EXPOSE $PORT

# run server
CMD exec uvicorn app.main:app --host 0.0.0.0 --port $PORT
