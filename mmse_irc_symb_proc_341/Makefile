CXX = icpx

ifeq ($(spr),1)
OUTPUT=spr_test
CPP_FLAG = -D_BBLIB_AVX512_ -D_BBLIB_SPR_ -march=sapphirerapids -O3 -DNDEBUG -std=c++17 -lpthread -mintrinsic-promote -Wno-c++11-narrowing -Wall -D_GNU_SOURCE -lstdc++
else
OUTPUT=icx_test
CPP_FLAG = -D_BBLIB_AVX512_ -xicelake-server -O3 -DNDEBUG -std=c++17 -lpthread -mintrinsic-promote -Wno-c++11-narrowing -Wall -D_GNU_SOURCE -lstdc++
endif

SRC += ./main.cpp\
       src/phy_matrix_inv_lemma.cpp\
       src/phy_matrix_inv_cholesky.cpp\
       src/phy_tafo_table_gen.cpp\
       src/phy_rx_mimo_mmse_avx512.cpp\
       src/phy_pusch_symbol_processing_5gnr.cpp\
       src/phy_pusch_symbol_processing_5gnr_avx512.cpp\
       src/phy_pusch_irc_symbol_processing_5gnr_avx512.cpp\
       src/phy_pusch_irc_symbol_processing_5gnr.cpp

INC += src/
OBJS = $(SRC:.cpp=.o)



all: $(OUTPUT)



$(OUTPUT): $(OBJS)
	$(CXX) $(LDFLAGS) $(CPP_FLAG) $(OBJS) -o $@



.cpp.o:
	$(CXX) -I$(INC) $(CPP_FLAG) -c $< -o $@



clean:
	rm -rf $(OBJS) $(OUTPUT) ./src/*.tmp
