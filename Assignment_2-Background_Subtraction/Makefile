CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

bg_sub: bg_sub.o mog.o mog.h
	g++ -O3 -std=c++11 $(CFLAGS) -o $@ $^ $(LIBS)
%.o: %.cpp
	g++ -O3 -std=c++11 $(CFLAGS) -c $< $(LIBS)

clean:
	rm -f *.o bg_sub
