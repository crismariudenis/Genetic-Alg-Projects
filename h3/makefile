
SHELL := $(shell if command -v fish > /dev/null 2>&1; then echo /usr/bin/fish; else echo /bin/bash; fi)
all:
	g++ main.cpp -O3 -std=c++20 -o  main && time stdbuf -oL ./main | tee -a logs

not_log:
	g++ main.cpp -O3 -std=c++20 -o main && time stdbuf -oL ./main

sa:
	g++ SA.cpp -O3 -std=c++20 -o sax && time stdbuf -oL ./sax | tee -a salogsx



clean:
	$(RM) main
