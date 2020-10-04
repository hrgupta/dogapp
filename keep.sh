#!/bin/bash
<<<<<<< HEAD
# starts the traffic app and restarts it if crashed

while true; do
    [ -e stopme ] && break
    streamlit run dogapp/dog.py --server.enableCORS False --server.port 8502
=======
while true; do
	[-e stopme] && break
	streamlit run dogapp/dog.py --server.enableCORS False --server.port 8502
>>>>>>> add restart script and warning on main app
done
