FROM ubuntu

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev

RUN apt-get install -y apt-utils

# install xfoil
RUN apt-get install -y xfoil
# install Xvfb
RUN apt-get install -y xvfb

RUN apt-get install -y firefox

# install python3
RUN apt-get install -y python3

# install pip
RUN apt-get install -y python3-pip
# copy project
COPY . /app
WORKDIR /app
# install dependencies
RUN pip install -r requirements.txt
# entrypoint script
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
