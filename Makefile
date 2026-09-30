CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall
TARGET   := convolution

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDLIBS := -framework OpenCL
else
    LDLIBS := -lOpenCL
endif

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

plots:
	python3 scripts/plot_results.py

clean:
	rm -f $(TARGET)

.PHONY: run plots clean
