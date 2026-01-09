CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -pedantic

TARGET := split_operator
SRC    := main.cpp

# Try to auto-detect Eigen include directory
EIGEN_CANDIDATES := \
  /usr/include/eigen3 \
  /usr/local/include/eigen3 \
  /opt/homebrew/include/eigen3 \
  $(CONDA_PREFIX)/include/eigen3 \
  $(CONDA_PREFIX)/include \
  $(EBROOTEIGEN)/include/eigen3 \
  $(EBROOTEIGEN)/include \
  $(EIGEN3_INCLUDE_DIR) \
  $(EIGEN_DIR) \
  .

# Pick the first directory that contains Eigen/Sparse
EIGEN_INC := $(firstword $(foreach d,$(EIGEN_CANDIDATES),$(if $(wildcard $(d)/Eigen/Sparse),$(d),)))

ifeq ($(strip $(EIGEN_INC)),)
$(error Could not find Eigen headers. Try: make EIGEN_DIR=/path/to/eigen3 (the folder that contains Eigen/))
endif

INCLUDES := -I$(EIGEN_INC)

all: $(TARGET)

$(TARGET): $(SRC) grid.h propagate.h
	@echo "Using Eigen include: $(EIGEN_INC)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

