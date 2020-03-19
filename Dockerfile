FROM python:3.5.5
WORKDIR /SSD_Flask_new

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD flask run