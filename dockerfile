FROM cnstark/pytorch:2.0.1-py3.9.17-ubuntu20.04

WORKDIR /workspace/

ADD requirements_docker.txt .
RUN pip install -r requirements_docker.txt

COPY billnerre ./billnerre

WORKDIR /workspace/billnerre/

#RUN chmod +x ./run.sh
#CMD ["bash","run.sh"]

CMD ["uvicorn", "api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
