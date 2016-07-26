CXX = clang++
# CXXFLAGS = -Wall -Wextra -std=c++11 -O2
# CXXFLAGS = -ggdb -Wall -Wextra -std=c++14 -I`root-config --incdir`
CXXFLAGS = -O3 -march=native -std=c++14 -I`root-config --incdir`
LDFLAGS = -larmadillo `root-config --libs` -lEG
# LDFLAGS =

OBJ = obj/start.o obj/MNISTReader.o obj/NeuralNetwork.o obj/GeneticAlgorithm.o

run: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# view: src.cpp
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm obj/*.o
