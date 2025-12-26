NVCC        := nvcc
NVCC_FLAGS  := -O3 -Xnvlink=-w
LIBS        := -lnvidia-ml -lpthread
TARGET      := pcie_diag
SRC         := pcie_diagnostic_pro.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)




