# run frontend
cd frontend/
ng serve &

# run backend
cd ../backend/
nodemon server.js &

# run deepfake engine
cd ../engines/fr/
bash run.sh &

# Keep the script running
tail -f /dev/null