#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/your/file.txt"
  exit 1
fi

FILE="./$1"

SERVERS=("client1_ip" "client2_ip" "client3_ip" "client4_ip" "client5_ip")
#FILE="./client.py"
DEST_PATH="/Path to client.py file on clients/$1"

for i in "${!SERVERS[@]}"; do
  server="${SERVERS[$i]}"
  echo "Deploying $DEST_PATH  on  $server with arg: $i"
  ssh -f "USERNAME@$server" "bash -c ' source /Path to Venv Activate && nohup /Python Path $DEST_PATH $i >> /Log Path  2>&1 &'" 
done
