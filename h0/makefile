
SHELL := $(shell if command -v fish > /dev/null 2>&1; then echo /usr/bin/fish; else echo /bin/bash; fi)

run:
	g++ main.cpp -O3 -std=c++20 -g0 -o main && time stdbuf -oL ./main | tee -a logs
clean:
	$(RM) main
