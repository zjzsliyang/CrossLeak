#!/bin/bash

while true; do
	./wifi_scanner.py &
	export scanner_pid=$!
	./pimotion.py &
	export camera_pid=$!
	./piaudio.py > /dev/null 2>&1 &
	export audio_pid=$!

	sleep 60
	kill $scanner_pid
	kill $camera_pid
	kill $audio_pid
	echo("time up")

	./renaming.py
	./file_transfer.py

done
