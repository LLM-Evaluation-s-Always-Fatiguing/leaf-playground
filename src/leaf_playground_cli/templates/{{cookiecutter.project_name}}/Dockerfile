FROM python:bullseye

RUN pip install leaf-playground=={{ cookiecutter.leaf_version }}

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["leaf-out"]

CMD ["version"]