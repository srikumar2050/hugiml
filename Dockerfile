FROM python:3.9.18
RUN apt-get clean && apt-get -y update
RUN apt-get -y install nano 
RUN apt-get -y install python3-dev && apt-get -y install build-essential

#install java
RUN apt-get -y install wget 
RUN mkdir -p /opt/java
RUN wget --no-cookies --no-check-certificate --header "Cookie: oraclelicense=accept-securebackup-cookie" https://download.oracle.com/java/22/latest/jdk-22_linux-x64_bin.tar.gz
RUN tar -xf jdk-22_linux-x64_bin.tar.gz -C /opt/java
RUN update-alternatives --install /usr/bin/java java /opt/java/jdk-22.0.1/bin/java 100
RUN update-alternatives --config java | grep 0

#copy files
COPY . /opt/hugiml/

#install python packages
ENV JUPYTER_ENABLE_LAB=yes
RUN pip install --upgrade pip
RUN pip install -r /opt/hugiml/requirements.txt
RUN rm /opt/hugiml/requirements.txt

#jupyter lab port 3333
EXPOSE 3333
CMD ["jupyter", "lab", "--notebook-dir=/opt/hugiml", "--ip='*'", "--port=3333", "--NotebookApp.token=''", \
                        "--no-browser", "--allow-root"]
