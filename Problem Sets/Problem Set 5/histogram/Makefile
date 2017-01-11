# Ernest Yeung
# ernestyalumni@gmail.com
# 20170110
NVCC = nvcc
NVCC_FLAGS = -std=c++11 -D_MWAITXINTRIN_H_INCLUDED

methodobjects  = histogram.o

all: main.exe

main.exe: main.o $(methodobjects)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	
main.o : main.cu $(methodobjects)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


# methods 

histogram.o : ./histogram/histogram.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@



clean:
	rm -f *.o *.exe
