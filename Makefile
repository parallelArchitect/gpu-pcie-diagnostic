NVCC        := nvcc
NVCC_FLAGS := -O3
LIBS       := -lnvidia-ml -pthread
TARGET     := pcie_diag
SRC        := pcie_diagnostic_pro.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)


