#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/your/file.txt"
  exit 1
fi

FILE="./$1"

SERVERS=("client1_ip" "client2_ip" "client3_ip" "client4_ip" "client5_ip")
#FILE="./client.py"
DEST_PATH="/Path/to/destination/on clients$1"

for server in "${SERVERS[@]}"; do
  echo "Copying to $server..."
  scp "$FILE" "USERNAME@$server:$DEST_PATH"
done
