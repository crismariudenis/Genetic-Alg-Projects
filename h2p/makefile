
SHELL := $(shell if command -v fish > /dev/null 2>&1; then echo /usr/bin/fish; else echo /bin/bash; fi)
all:
	g++ main.cpp -O3 -std=c++20 -DPOWER=3 -DGEN=0.5 -o main && time stdbuf -oL ./main | tee -a logs


run_bash:
	SHELL := /bin/bash
	for g in 0.2 0.4 0.6 0.8 0.9 1.0; do \
        for p in 0.1 0.3 0.5 0.6 0.7 0.8 0.9 1.0 1.3 1.5 1.6 2.0 2.1 2.2 2.5 2.7 3.0; do \
			echo "p: $$p, g: $$g"; \
			g++ main.cpp -O3 -std=c++20 -DPOWER=$$p -DGEN=$$g -o main && time stdbuf -oL ./main | tee -a logs; \
		done; \
	done

clean:
	$(RM) main
